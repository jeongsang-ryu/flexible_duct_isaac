"""Shift+drag a hoop, using the force path that is LEGAL under direct-GPU API.

WHY NOT JUST USE THE BUILT-IN GRAB. omni.physx.ui's grab calls
PxRigidDynamic::addForce(). PhysX refuses that outright once
PxSceneFlag::eENABLE_DIRECT_GPU_API is set:

    PxRigidDynamic::addForce(): it is illegal to call this method
    if PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled!

and NVIDIA's own code states that "setting the device to cuda enables fabric
and PhysX direct-GPU API". Cloth is GPU-only, so cuda is not optional here:
using fabric at all rules the built-in grab out. Enabling omni.physx.ui makes
the gesture fire, but every frame of it errors instead of moving anything.

isaacsim.core.prims.RigidPrim.apply_forces goes through the tensor API, which
IS the supported route under that flag. So this re-implements the gesture on
top of it: pick the hoop nearest the click ray, then each frame pull it toward
the cursor with a damped spring.

    from duct_sim.mouse_drag import HoopDragger
    dragger = HoopDragger(ring_paths)      # after sim.play()
    ...
    dragger.update()                       # once per step, before sim.step()
"""

from __future__ import annotations

import carb
import carb.input
import numpy as np
import omni.appwindow
import omni.usd
from pxr import Gf, UsdGeom


def _to_backend(arr):
    """numpy -> whatever the core backend wants.

    With a GPU pipeline Isaac switches its backend from numpy to torch ("NumPy
    cannot be used with GPU pipelines"), and apply_forces then calls .to() on
    whatever it is handed -- so a numpy array fails with
    "'numpy.ndarray' object has no attribute 'to'".
    """
    try:
        import torch
        from isaacsim.core.simulation_manager import SimulationManager
        try:
            dev = SimulationManager.get_physics_sim_device()
        except Exception:
            dev = "cuda"
        return torch.as_tensor(arr, dtype=torch.float32, device=dev)
    except Exception:
        return arr


class HoopDragger:
    def __init__(self, ring_paths, stiffness=60.0, damping=6.0, max_force=40.0):
        self.ring_paths = list(ring_paths)
        self.k = float(stiffness)
        self.c = float(damping)
        self.max_force = float(max_force)

        self._view = None
        self._held = None          # index into ring_paths
        self._depth = 1.0          # distance from eye to the grabbed hoop at pick time
        self._input = carb.input.acquire_input_interface()
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._mouse = omni.appwindow.get_default_app_window().get_mouse()
        self._was_down = False
        self._pick_pending = False
        self._target = None        # world-space goal, moved by mouse DELTAS
        self._last_mouse = None
        self._saw_shift = False
        self._saw_mouse = False

    # -- lazily build the view: RigidPrim allocates GPU tensors, so do it after play
    def _ensure_view(self):
        if self._view is None:
            from isaacsim.core.prims import RigidPrim
            self._view = RigidPrim(self.ring_paths)
            # INITIALIZE, or every pose reads back as the origin. Without this
            # get_world_poses() returned (0,0,0) for all 819 hoops, so every
            # hoop projected to the same pixel and the pick always landed on
            # index 0 -- which is exactly what "it grabs the wrong hoop" was,
            # for both the ray version and the screen-space version. Two
            # rewrites went into the coordinate maths for a bug that was never
            # in the coordinate maths.
            try:
                self._view.initialize()
            except Exception as exc:
                print(f"[drag] view initialize failed: {exc}", flush=True)

            # and CHECK it took, rather than assume
            p = self._positions()
            spread = float(np.ptp(p, axis=0).max()) if len(p) else 0.0
            if spread < 1e-6:
                print(f"[drag] WARNING hoop poses have zero spread "
                      f"({len(p)} hoops all at the same point) -- picking "
                      f"cannot work", flush=True)
            else:
                print(f"[drag] hoop poses span {spread:.2f} m", flush=True)
        return self._view

    def _shift_down(self):
        for dev in (self._kb, None):
            try:
                if (self._input.get_keyboard_value(dev, carb.input.KeyboardInput.LEFT_SHIFT)
                        or self._input.get_keyboard_value(dev, carb.input.KeyboardInput.RIGHT_SHIFT)):
                    return True
            except Exception:
                continue
        return False

    def _mouse_down(self):
        # PASS None, NOT THE MOUSE HANDLE. Kit's own code polls
        # get_mouse_value(None, ...) and querying with the app window's mouse
        # object returned 0 forever here -- the gesture was never detected, no
        # error, nothing in the log. The handle form is kept as a fallback.
        for dev in (None, self._mouse):
            try:
                if self._input.get_mouse_value(dev, carb.input.MouseInput.LEFT_BUTTON):
                    return True
            except Exception:
                continue
        return False

    # ---- exact pick: ask the viewport what is under the cursor ----
    # Computing a world ray from carb's mouse coords is off by a constant,
    # because get_mouse_coords_normalized normalizes to the WINDOW while the 3D
    # viewport is only a sub-rectangle of it (toolbars left and top, panels
    # right and bottom). That offset is exactly the "force lands slightly away
    # from where I clicked" symptom. The viewport's own query does the pick on
    # the GPU in its own space, so it cannot drift.
    def _request_pick(self):
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            if vp is None or not hasattr(vp, "request_query"):
                return False
            mx, my = self._input.get_mouse_coords_pixel(self._mouse)

            def _on_query(path, pixel, *_):
                if not path:
                    return
                sp = str(path)
                for i, rp in enumerate(self.ring_paths):
                    # the hit is on a capsule child, so match by prefix
                    if sp == rp or sp.startswith(rp + "/"):
                        self._held = i
                        self._pick_pending = False
                        pos = self._positions()[i]
                        eye = self._eye()
                        self._depth = (float(np.linalg.norm(pos - eye))
                                       if eye is not None else 1.0)
                        print(f"[drag] picked {rp}", flush=True)
                        return

            vp.request_query((int(mx), int(my)), _on_query)
            return True
        except Exception:
            return False

    def _screen_positions(self):
        """Hoop centres in WINDOW pixels, via the viewport's own projection.

        Building a world ray from carb's normalized mouse coords cannot be made
        to work: those are normalized to the WINDOW, while the 3D viewport is a
        sub-rectangle of it (toolbars left and top, panels right and bottom).
        The ray is therefore offset by however large those panels are, measured
        as a 23-51 cm miss at the duct, and it always grabbed whichever hoop
        happened to sit near the centre of that offset.

        Project the other way instead: world -> NDC through
        viewport_api.world_to_ndc, then NDC -> window pixels through the frame
        rectangle. That is exactly what omni.kit.viewport.utility's
        get_ui_position_for_prim does, so the result lands in the same space
        carb reports the cursor in, and no focal length, aperture or panel size
        has to be guessed.
        """
        try:
            import omni.ui
            from omni.kit.viewport.utility import get_active_viewport_window
            win = get_active_viewport_window()
            if win is None:
                return None
            vp = win.viewport_api
            mvp = vp.world_to_ndc
            frame = win.frame

            dpi = omni.ui.Workspace.get_dpi_scale() or 1.0
            splitter = 4 * dpi                      # kit's dock splitter
            tab_h = 0
            if win.dock_tab_bar_visible or not (win.flags & omni.ui.WINDOW_FLAGS_NO_TITLE_BAR):
                tab_h = 22 * dpi

            out = np.full((len(self.ring_paths), 2), np.inf, dtype=np.float64)
            for i, p in enumerate(self._positions()):
                ndc = mvp.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                if ndc[2] < 0:                      # behind the camera
                    continue
                x = (ndc[0] + 1.0) * 0.5
                y = 1.0 - (ndc[1] + 1.0) * 0.5
                out[i, 0] = dpi * (frame.screen_position_x + x * frame.computed_width) - splitter
                out[i, 1] = (dpi * (frame.screen_position_y + y * frame.computed_height)
                             - tab_h - splitter)
            return out
        except Exception as exc:
            if not getattr(self, "_warned_screen", False):
                self._warned_screen = True
                print(f"[drag] screen projection unavailable ({exc}); "
                      f"falling back to the ray", flush=True)
            return None

    def _eye(self):
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            stage = omni.usd.get_context().get_stage()
            cam = stage.GetPrimAtPath(vp.camera_path)
            m = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0)
            return np.array(m.ExtractTranslation(), dtype=np.float64)
        except Exception:
            return None

    def _camera_basis(self):
        """(right, up) unit vectors of the camera, in world space."""
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            stage = omni.usd.get_context().get_stage()
            cam = stage.GetPrimAtPath(vp.camera_path)
            m = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0)
            r = m.ExtractRotationMatrix()
            right = np.array(Gf.Vec3d(1, 0, 0) * r, dtype=np.float64)
            up = np.array(Gf.Vec3d(0, 1, 0) * r, dtype=np.float64)
            return right / np.linalg.norm(right), up / np.linalg.norm(up)
        except Exception:
            return None, None

    def _camera_ray(self):
        """Ray through the cursor, in world space. None if it cannot be built."""
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            if vp is None:
                return None
            stage = omni.usd.get_context().get_stage()
            cam = stage.GetPrimAtPath(vp.camera_path)
            if not cam or not cam.IsValid():
                return None
            xform = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0)
            eye = np.array(xform.ExtractTranslation(), dtype=np.float64)

            mx, my = self._input.get_mouse_coords_normalized(self._mouse)
            # normalized coords are 0..1 from the top-left of the window
            ndc_x = mx * 2.0 - 1.0
            ndc_y = 1.0 - my * 2.0

            gcam = UsdGeom.Camera(cam)
            focal = gcam.GetFocalLengthAttr().Get() or 50.0
            hap = gcam.GetHorizontalApertureAttr().Get() or 20.955
            vap = gcam.GetVerticalApertureAttr().Get() or 15.2908

            # camera looks down -Z in its own space
            dir_cam = Gf.Vec3d(ndc_x * hap * 0.5 / focal,
                               ndc_y * vap * 0.5 / focal,
                               -1.0)
            rot = xform.ExtractRotationMatrix()
            d = rot.GetTranspose() * dir_cam if False else dir_cam * rot
            d = np.array(d, dtype=np.float64)
            n = np.linalg.norm(d)
            if n < 1e-9:
                return None
            return eye, d / n
        except Exception:
            return None

    def _positions(self):
        view = self._ensure_view()
        pos, _ = view.get_world_poses()
        return np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos, dtype=np.float64)

    def _velocities(self):
        view = self._ensure_view()
        try:
            vel = view.get_velocities()
            vel = np.asarray(vel.cpu() if hasattr(vel, "cpu") else vel, dtype=np.float64)
            return vel[:, :3]
        except Exception:
            return np.zeros((len(self.ring_paths), 3))

    def update(self):
        """Call once per simulation step, before sim.step()."""
        shift, mouse = self._shift_down(), self._mouse_down()
        # say so the FIRST time each is seen, so "nothing happens" can be told
        # apart from "shift was read but the click was not"
        if shift and not self._saw_shift:
            self._saw_shift = True
            print("[drag] shift detected", flush=True)
        if mouse and not self._saw_mouse:
            self._saw_mouse = True
            print("[drag] left button detected", flush=True)
        down = shift and mouse

        if down and self._held is None:
            # ALWAYS take the ray pick, immediately. The previous version tried
            # the viewport query first and only fell back "if unavailable" --
            # but request_query exists and accepts the call, then never matches,
            # because it wants VIEWPORT TEXTURE pixels while carb hands out
            # WINDOW pixels. So the pick stayed pending forever and no force was
            # ever applied. The query is now only a refinement on top of a pick
            # that has already succeeded.
            # FIRST: pick in screen space. This is the one that is correct --
            # it uses the viewport's own projection, so it needs no assumption
            # about where the viewport sits inside the window.
            scr = self._screen_positions()
            if scr is not None:
                mx, my = self._input.get_mouse_coords_pixel(self._mouse)
                d = np.linalg.norm(scr - np.array([mx, my], dtype=np.float64), axis=1)
                if not getattr(self, "_dumped", False):
                    # ONE-SHOT DIAGNOSTIC. The first screen-space attempt still
                    # grabbed a hoop 535 px away and always index 0, which says
                    # the projection is landing somewhere wrong rather than
                    # being slightly off. Print the spaces involved instead of
                    # guessing which one is at fault.
                    self._dumped = True
                    fin = np.isfinite(scr[:, 0])
                    print(f"[drag/dbg] cursor px=({mx:.0f}, {my:.0f})", flush=True)
                    print(f"[drag/dbg] hoops finite {int(fin.sum())}/{len(scr)}",
                          flush=True)
                    if fin.any():
                        f = scr[fin]
                        print(f"[drag/dbg] projected x {f[:,0].min():.0f}..{f[:,0].max():.0f}  "
                              f"y {f[:,1].min():.0f}..{f[:,1].max():.0f}", flush=True)
                        print(f"[drag/dbg] nearest {d[np.argmin(d)]:.0f} px, "
                              f"farthest {np.nanmax(d[np.isfinite(d)]):.0f} px", flush=True)
                    try:
                        import omni.ui
                        from omni.kit.viewport.utility import get_active_viewport_window
                        w = get_active_viewport_window()
                        fr = w.frame
                        print(f"[drag/dbg] frame screen=({fr.screen_position_x:.0f}, "
                              f"{fr.screen_position_y:.0f}) size=({fr.computed_width:.0f}, "
                              f"{fr.computed_height:.0f}) dpi={omni.ui.Workspace.get_dpi_scale():.2f} "
                              f"tab={w.dock_tab_bar_visible}", flush=True)
                        print(f"[drag/dbg] viewport resolution={w.viewport_api.resolution}",
                              flush=True)
                    except Exception as exc:
                        print(f"[drag/dbg] frame unavailable: {exc}", flush=True)
                i = int(np.argmin(d))
                if np.isfinite(d[i]):
                    self._held = i
                    eye = self._eye()
                    pos = self._positions()[i]
                    self._depth = (float(np.linalg.norm(pos - eye))
                                   if eye is not None else 1.0)
                    print(f"[drag] grabbed {self.ring_paths[i]} "
                          f"({d[i]:.0f} px from the cursor)", flush=True)

            ray = self._camera_ray() if self._held is None else None
            if ray is not None:
                eye, d = ray
                pos = self._positions()
                rel = pos - eye
                t = rel @ d
                perp = np.linalg.norm(rel - np.outer(t, d), axis=1)
                perp[t < 0] = np.inf
                i = int(np.argmin(perp))
                if np.isfinite(perp[i]):
                    # no distance threshold: a click always grabs the nearest
                    # hoop, so the gesture never silently does nothing
                    self._held = i
                    self._depth = float(t[i])
                    print(f"[drag] grabbed {self.ring_paths[i]} "
                          f"({perp[i]*100:.1f} cm off the click ray)", flush=True)
            if self._held is None:
                # last resort: nearest hoop to the camera
                pos = self._positions()
                eye = self._eye()
                if eye is not None and len(pos):
                    i = int(np.argmin(np.linalg.norm(pos - eye, axis=1)))
                    self._held = i
                    self._depth = float(np.linalg.norm(pos[i] - eye))
                    print(f"[drag] ray unavailable; grabbed nearest hoop "
                          f"{self.ring_paths[i]}", flush=True)

        if not down:
            if self._held is not None:
                print(f"[drag] released {self.ring_paths[self._held]}", flush=True)
            self._held = None
            self._pick_pending = False
            self._target = None
            self._last_mouse = None
            return

        if self._held is None:
            return

        # --- move the target by mouse DELTAS, not absolute ray position ---
        # Deltas cancel any constant offset between window space and viewport
        # space, so the hoop follows the cursor's MOTION exactly even if the
        # absolute mapping is imperfect.
        mx, my = self._input.get_mouse_coords_normalized(self._mouse)
        pos = self._positions()[self._held]
        if self._target is None:
            self._target = pos.copy()
            self._last_mouse = (mx, my)

        right, up = self._camera_basis()
        if right is not None:
            dx = mx - self._last_mouse[0]
            dy = my - self._last_mouse[1]
            # screen-space motion scaled to world by the depth of the hoop
            scale = 2.0 * self._depth * 0.9
            self._target = self._target + right * (dx * scale) - up * (dy * scale)
        self._last_mouse = (mx, my)

        vel = self._velocities()[self._held]
        f = self.k * (self._target - pos) - self.c * vel
        mag = float(np.linalg.norm(f))
        if mag > self.max_force:
            f = f / mag * self.max_force

        forces = np.zeros((len(self.ring_paths), 3), dtype=np.float32)
        forces[self._held] = f
        try:
            self._ensure_view().apply_forces(_to_backend(forces), is_global=True)
        except Exception as exc:
            print(f"[drag] apply_forces failed: {exc}", flush=True)
            self._held = None
