#!/usr/bin/env bash
# record_gui.sh <python-script> [args...] -- run a GUI Isaac scene and film it.
#
# WHY SCREEN-CAPTURE INSTEAD OF RENDERING FRAMES.
# The headless replicator path reads the USD stage, but PhysicsContext turns
# Fabric on whenever the device is CUDA, and simulated transforms and deformed
# points then live in Fabric and are never written back to USD. So headless
# capture shows an empty scene. Disabling Fabric puts the results back on the
# stage and the render fills in -- but the motion comes out visibly wrong (a
# duct dropped from 2.2 m took 8 s to fall 1.5 m instead of landing in 0.64 s),
# because the writeback does not keep up with the simulation.
#
# The GUI viewport reads Fabric directly and has been correct the whole time.
# Filming the viewport therefore records what the simulation is ACTUALLY doing,
# with no writeback in the path -- the physics stays untouched and only the
# capture method changes.
set -uo pipefail

SCRIPT="${1:?usage: record_gui.sh <python-script> [args...]}"
shift
OUT="${OUT:-/tmp/gui_capture.mp4}"
SECONDS_TO_RECORD="${SECONDS_TO_RECORD:-25}"
BOOT_WAIT="${BOOT_WAIT:-180}"
# Crop to the 3D viewport. A full-desktop grab also films the terminal, the
# stage tree and whatever notification happens to pop up, and shrinks the
# subject to a corner of the frame.
CROP="${CROP:-1000x640+128+96}"
SETTLE="${SETTLE:-1}"
DISP="${DISPLAY:-:1}"
# NESTED DISPLAY. The desktop on :1 is the user's live session -- Slack, Discord
# and notifications kept landing on top of the viewport mid-capture, and half
# the frames came out showing a chat window. Xephyr gives Isaac its own X
# server, so the grab can only ever contain Isaac. Set NESTED=0 to film the
# real desktop instead.
NESTED="${NESTED:-1}"
NESTED_DISP="${NESTED_DISP:-:7}"
NESTED_SIZE="${NESTED_SIZE:-1280x800}"
XEPHYR_PID=""
if [ "$NESTED" = "1" ]; then
  Xephyr "$NESTED_DISP" -screen "$NESTED_SIZE" -resizeable -ac \
         >/tmp/rec_xephyr.log 2>&1 &
  XEPHYR_PID=$!
  for _ in $(seq 20); do
    [ -e "/tmp/.X11-unix/X${NESTED_DISP#:}" ] && break
    sleep 0.5
  done
  if [ -e "/tmp/.X11-unix/X${NESTED_DISP#:}" ]; then
    DISP="$NESTED_DISP"
    CROP="${CROP_NESTED:-820x460+53+33}"   # the 3D viewport inside the Kit window
    echo "[rec] nested X on $NESTED_DISP ($NESTED_SIZE)"
  else
    echo "[rec] Xephyr did not come up; filming the real desktop"
    XEPHYR_PID=""
  fi
fi

cd /home/js/hmcl_issac_project/duct_sim
source /home/js/miniforge3/etc/profile.d/conda.sh
conda activate hmclab6

echo "[rec] launching $SCRIPT $*"
DISPLAY="$DISP" setsid nohup python "$SCRIPT" "$@" > /tmp/rec_scene.log 2>&1 &
SCENE_PID=$!

echo "[rec] waiting ${BOOT_WAIT}s for the window (first boot compiles shaders)"
# Start filming the INSTANT the scene reports it is running. The previous
# version waited a further 8 s, by which time a 2.2 m drop (0.64 s of fall) was
# long over and every frame showed the duct already settled.
for _ in $(seq "$BOOT_WAIT"); do
  if grep -qE 'running\.' /tmp/rec_scene.log 2>/dev/null; then break; fi
  sleep 1
done
sleep "$SETTLE"

echo "[rec] filming ${SECONDS_TO_RECORD}s of $DISP -> $OUT"
ffmpeg -y -f x11grab -framerate 30 -video_size "${CROP%%+*}" \
       -i "$DISP+${CROP#*+}" \
       -t "$SECONDS_TO_RECORD" -vf "scale=900:-2" \
       -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT" 2>/tmp/rec_ffmpeg.log

echo "[rec] stopping the scene"
kill "$SCENE_PID" 2>/dev/null
pkill -f "$(basename "$SCRIPT")" 2>/dev/null
[ -n "$XEPHYR_PID" ] && kill "$XEPHYR_PID" 2>/dev/null

ls -l "$OUT" 2>/dev/null || echo "[rec] NO OUTPUT -- check /tmp/rec_ffmpeg.log"
