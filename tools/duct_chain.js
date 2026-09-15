/* Duct as a fixed-length chain that routes around posts.
 *
 * Kept in its own file so the planner and the test harness run the SAME code.
 * The first version of this lived inside the page and shipped with a bug that
 * only shows up when you drag: it pushed NODES out of a post but left the
 * SEGMENTS between them free to cut straight through. With 0.5 m nodes and a
 * 0.25 m post, a post fits entirely between two nodes, so the duct sailed
 * through it while every node was legitimately outside.
 *
 * Everything here is pure: no DOM, no canvas.
 */
(function (root) {
  "use strict";

  function makeDuct(lengthM, nodeSpacing, cx, cy, name) {
    const n = Math.max(2, Math.round(lengthM / nodeSpacing) + 1);
    const seg = lengthM / (n - 1);
    const pts = [];
    for (let i = 0; i < n; i++) pts.push([cx - lengthM / 2 + seg * i, cy]);
    return { name: name || "duct", seg, pts };
  }

  function chainLength(r) { return r.seg * (r.pts.length - 1); }

  /* Closest point on segment ab to p, as the parameter t in [0,1]. */
  function closestT(ax, ay, bx, by, px, py) {
    const dx = bx - ax, dy = by - ay;
    const L2 = dx * dx + dy * dy;
    if (L2 < 1e-12) return 0;
    return Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L2));
  }

  /* Push the duct out of every post.
   *
   * Works on SEGMENTS, not just nodes, so a post smaller than the node spacing
   * cannot slip between two nodes. The clearance is the post radius plus the
   * duct's own radius, so it is the duct's SURFACE that clears the post rather
   * than its centreline.
   */
  function pushOutOfPosts(pts, posts, ductRadius, held) {
    if (!posts || !posts.length) return;
    for (let k = 0; k < posts.length; k++) {
      const a = posts[k];
      const R = a.r + ductRadius;

      for (let i = 0; i < pts.length - 1; i++) {
        const p = pts[i], q = pts[i + 1];
        const t = closestT(p[0], p[1], q[0], q[1], a.x, a.y);
        const cx = p[0] + (q[0] - p[0]) * t;
        const cy = p[1] + (q[1] - p[1]) * t;
        let nx = cx - a.x, ny = cy - a.y;
        let d = Math.hypot(nx, ny);
        if (d >= R) continue;
        if (d < 1e-9) { nx = 1; ny = 0; d = 1e-9; }
        nx /= d; ny /= d;
        const push = R - d;
        // share the correction between the endpoints by how close each is to
        // the contact; iterating converges on the contact point itself moving
        // the full distance
        const wp = 1 - t, wq = t;
        if (i !== held) { p[0] += nx * push * wp; p[1] += ny * push * wp; }
        if (i + 1 !== held) { q[0] += nx * push * wq; q[1] += ny * push * wq; }
      }

      // endpoints can still end up inside on their own
      for (let i = 0; i < pts.length; i++) {
        if (i === held) continue;
        let nx = pts[i][0] - a.x, ny = pts[i][1] - a.y;
        let d = Math.hypot(nx, ny);
        if (d >= R) continue;
        if (d < 1e-9) { nx = 1; ny = 0; d = 1e-9; }
        pts[i][0] = a.x + nx / d * R;
        pts[i][1] = a.y + ny / d * R;
      }
    }
  }

  /* Re-place every node so each gap is EXACTLY the rest length, walking out
   * from an anchor node in both directions.
   *
   * The first solver used symmetric Jakobsen relaxation and did not converge
   * on a hard drag -- a 6.000 m duct measured 6.231 m -- and with a post driven
   * through it, it diverged outright. Walking out from an anchor is exact in a
   * single pass and cannot blow up, because each node is simply placed at a
   * fixed distance from the previous one.
   */
  function enforceLengths(pts, L, anchor) {
    for (let i = anchor + 1; i < pts.length; i++) {
      let dx = pts[i][0] - pts[i - 1][0], dy = pts[i][1] - pts[i - 1][1];
      let d = Math.hypot(dx, dy);
      if (d < 1e-9) { dx = 1; dy = 0; d = 1; }
      pts[i][0] = pts[i - 1][0] + dx / d * L;
      pts[i][1] = pts[i - 1][1] + dy / d * L;
    }
    for (let i = anchor - 1; i >= 0; i--) {
      let dx = pts[i][0] - pts[i + 1][0], dy = pts[i][1] - pts[i + 1][1];
      let d = Math.hypot(dx, dy);
      if (d < 1e-9) { dx = -1; dy = 0; d = 1; }
      pts[i][0] = pts[i + 1][0] + dx / d * L;
      pts[i][1] = pts[i + 1][1] + dy / d * L;
    }
  }

  /* Alternate "exact lengths" with "outside every post" until both hold.
   * Lengths go last, so the duct can never end a drag stretched. */
  function relax(r, posts, ductRadius, held, iters) {
    const pts = r.pts;
    const anchor = (held >= 0 && held < pts.length) ? held : (pts.length >> 1);
    const N = iters || 36;
    for (let it = 0; it < N; it++) {
      enforceLengths(pts, r.seg, anchor);
      pushOutOfPosts(pts, posts, ductRadius, held);
    }
    enforceLengths(pts, r.seg, anchor);
  }

  /* Smallest distance from any post centre to the duct's centreline. Used by
   * the tests to assert the duct really is outside. */
  function minPostDistance(pts, post) {
    let best = Infinity;
    for (let i = 0; i < pts.length - 1; i++) {
      const t = closestT(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], post.x, post.y);
      const cx = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t;
      const cy = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t;
      best = Math.min(best, Math.hypot(cx - post.x, cy - post.y));
    }
    return best;
  }

  root.DuctChain = { makeDuct, chainLength, relax, enforceLengths,
                     pushOutOfPosts, minPostDistance, closestT };
})(typeof window !== "undefined" ? window : globalThis);
