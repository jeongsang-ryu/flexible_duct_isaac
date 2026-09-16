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
    def __init__(self, ring_paths, stiffness=60.0, damping=6.0, max_force=40.0,
                 tie_band_px=3.0):
        self.ring_paths = list(ring_paths)
        self.k = float(stiffness)
        self.c = float(damping)
        self.max_force = float(max_force)
        # narrower than the on-screen hoop spacing (4-9 px), so the band
        # only ever holds hoops that genuinely overlap in the view
        self.tie_band_px = float(tie_band_px)

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
        self._saw_keys = set()
        self._warned_arm = False
        self._calib = 0
        self._last_frame = None
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

    # Keys that arm the drag. SHIFT is the documented one but it is a
    # modifier, and modifiers can be consumed before they reach carb -- the log
    # showed "left button detected" with no matching "shift detected", so the
    # click was arriving and the modifier was not. G is a plain letter and
    # cannot be swallowed the same way, so it is offered as an alternative
    # rather than a replacement.
    _ARM_KEYS = (("LEFT_SHIFT", "shift"), ("RIGHT_SHIFT", "shift"), ("G", "G"))

    def _shift_down(self):
        # device None FIRST. Kit polls the mouse as get_mouse_value(None, ...)
        # and passing the app window's handle returns 0 forever; the keyboard
        # behaves the same way, and trying the handle first hid that.
        for dev in (None, self._kb):
            for attr, label in self._ARM_KEYS:
                key = getattr(carb.input.KeyboardInput, attr, None)
                if key is None:
                    continue
                try:
                    if self._input.get_keyboard_value(dev, key):
                        if label not in self._saw_keys:
                            self._saw_keys.add(label)
                            print(f"[drag] armed by {label}", flush=True)
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

            # MEASURED, not assumed. Solving each pick backwards for the
            # frame size that would have put the grabbed hoop under the cursor
            # returned 1285x719 against a widget of 949x577 -- that is the
            # RENDER TEXTURE (1280x720). carb reports the cursor in texture
            # pixels, so the widget rectangle, its dpi scaling, the dock
            # splitter and the tab bar are all irrelevant here; they describe
            # where the viewport sits on screen, which is a different question.
            # Copying get_ui_position_for_prim was the mistake: that function
            # places UI overlays in window space, not mouse picks.
            res = vp.resolution
            res_w, res_h = float(res[0]), float(res[1])
            dpi = omni.ui.Workspace.get_dpi_scale() or 1.0

            self._last_frame = (res_w, res_h, 0.0, 0.0, dpi)
            out = np.full((len(self.ring_paths), 2), np.inf, dtype=np.float64)
            for i, p in enumerate(self._positions()):
                ndc = mvp.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                if ndc[2] < 0:                      # behind the camera
                    continue
                x = (ndc[0] + 1.0) * 0.5
                y = 1.0 - (ndc[1] + 1.0) * 0.5
                out[i, 0] = x * res_w
                out[i, 1] = y * res_h
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
        # _shift_down already names which key armed it, so this would only
        # repeat it. What is worth saying once is the opposite case: the click
        # arrives and nothing arms it, which is exactly what happened -- "left
        # button detected" with no arming key, and silence from there.
        if mouse and not shift and not self._saw_keys and not self._warned_arm:
            self._warned_arm = True
            print("[drag] click seen but no arming key -- hold SHIFT or G "
                  "while dragging", flush=True)
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
                # DEPTH BREAKS THE TIE. Nearest-in-pixels alone grabs whatever
                # projects closest, which on an overlapping run of hoops can be
                # one BEHIND the duct you clicked -- "sometimes it moves
                # something else". Among everything under the cursor, take the
                # one nearest the camera, which is the one actually visible
                # there. The radius is generous because a hoop is 0.4 m across
                # and the click lands on its surface, not its centre.
                # Depth may only separate candidates that are at essentially
                # the SAME pixel -- two ducts overlapping in the view. It must
                # not reorder neighbours along one duct, and a fixed 45 px
                # radius did exactly that: hoops sit 4.4 px apart at 16 m and
                # 8.9 px at 8 m, so 45 px spans five to ten of them and the
                # nearest-to-camera one wins from anywhere in that window.
                # Every pick then reported 43-45 px, sitting at the radius,
                # which is the fingerprint of the radius choosing rather than
                # the cursor.
                eye0 = self._eye()
                if eye0 is not None and len(d):
                    dmin = float(np.min(d))
                    if np.isfinite(dmin):
                        # a band narrower than one hoop spacing: only genuine
                        # co-located candidates get in
                        tied = d <= dmin + self.tie_band_px
                        if tied.sum() > 1:
                            depth = np.full(len(d), np.inf)
                            pos_all = self._positions()
                            depth[tied] = np.linalg.norm(pos_all[tied] - eye0, axis=1)
                            d = np.where(tied, depth, np.inf)
                i = int(np.argmin(d))
                if np.isfinite(d[i]):
                    self._held = i
                    eye = self._eye()
                    pos = self._positions()[i]
                    self._depth = (float(np.linalg.norm(pos - eye))
                                   if eye is not None else 1.0)
                    off = scr[i] - np.array([mx, my], dtype=np.float64)
                    # PER-PICK CALIBRATION. The offsets are not constant
                    # (-332..+339 in x), so this is not a fixed shift; it looks
                    # like the NDC->pixel SCALE is wrong. Report what the
                    # mapping was built from, and what scale would have put the
                    # picked hoop exactly under the cursor, so the right one can
                    # be read off instead of guessed at a seventh time.
                    if self._calib < 6:
                        self._calib += 1
                        c = self._last_frame
                        if c:
                            fw, fh, fx0, fy0, dpi = c
                            # solve for the width/height that makes scr[i] == cursor
                            nx = (scr[i][0] - fx0) / fw if fw else 0.0
                            ny = (scr[i][1] - fy0) / fh if fh else 0.0
                            need_w = (mx - fx0) / nx if abs(nx) > 1e-6 else float("nan")
                            need_h = (my - fy0) / ny if abs(ny) > 1e-6 else float("nan")
                            print(f"[calib] frame {fw:.0f}x{fh:.0f} at "
                                  f"({fx0:.0f},{fy0:.0f}) dpi {dpi:.2f} | "
                                  f"cursor ({mx:.0f},{my:.0f}) | "
                                  f"implied frame {need_w:.0f}x{need_h:.0f}",
                                  flush=True)
                    print(f"[drag] grabbed {self.ring_paths[i]} "
                          f"({np.linalg.norm(off):.0f} px from the cursor, "
                          f"offset ({off[0]:+.0f}, {off[1]:+.0f}), "
                          f"{self._depth:.1f} m from the camera)", flush=True)

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
