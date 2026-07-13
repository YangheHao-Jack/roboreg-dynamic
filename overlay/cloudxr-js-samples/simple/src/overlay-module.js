// overlay-module.js
// ---------------------------------------------------------------------------
// Drop-in robot-overlay renderer for the NVIDIA CloudXR.js `simple/` sample.
//
// CloudXR.js owns the WebXR session + WebGL2 context and composites the
// streamed eye buffers into the XRWebGLLayer framebuffer every frame. This
// module renders the URDF robot on TOP of that frame, into the same
// framebuffer, using the same per-eye views — so the on-device overlay lines
// up with the streamed scene with no second session and no capture round-trip.
//
// Wiring (in simple/src/main.ts):
//   import { OverlayModule } from './overlay-module.js';
//   const overlay = new OverlayModule();   // world-locked by default
//   new CloudXRClient({
//     onWebGLInitialized: (gl, _wq, referenceSpace) => {
//       overlay.init(gl, referenceSpace);
//       overlay.loadURDF('lbr_med7_r800.urdf');   // served next to the page
//       startPoseFeed(overlay);                    // rosbridge or data channel
//     },
//     onXRFrame: (t, frame, gl, baseLayer) => overlay.render(frame, baseLayer),
//   });
//
// Pose feed is intentionally external: call overlay.setPoseFromPosQuat(p,q) /
// overlay.setPoseFromArray(m16) and overlay.setJoints(positions, names) from
// whatever transport you use (rosbridge now, the CloudXR opaque data channel
// later). The module never knows or cares where poses come from.
//
// Requires (sample already uses webpack/npm):  npm i three urdf-loader
// ---------------------------------------------------------------------------

import * as THREE from 'three';
import URDFLoader from 'urdf-loader';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { ColladaLoader } from 'three/examples/jsm/loaders/ColladaLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// Pose convention (identical to the standalone viewer, verified against srbag):
//   poseRel = CV2GL * inv(H * HT_OPTICAL)
// where H is the raw camera_to_base (eye-in-base) extrinsic.
const HT_OPTICAL = new THREE.Matrix4().set(
  0, 0, 1, 0,
  -1, 0, 0, 0,
  0, -1, 0, 0,
  0, 0, 0, 1);
const CV2GL = new THREE.Matrix4().set(
  1, 0, 0, 0,
  0, -1, 0, 0,
  0, 0, -1, 0,
  0, 0, 0, 1);

const _eye = new THREE.Matrix4();

export class OverlayModule {
  constructor(opts = {}) {
    this.scene = new THREE.Scene();
    this.poseGroup = new THREE.Group();
    this.poseGroup.matrixAutoUpdate = false;
    this.scene.add(this.poseGroup);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x333844, 1.1));
    const d = new THREE.DirectionalLight(0xffffff, 1.4);
    d.position.set(2, 4, 3);
    this.scene.add(d);

    this.camera = new THREE.PerspectiveCamera();
    this.camera.matrixAutoUpdate = false;

    this.robot = null;
    this.poseRel = new THREE.Matrix4();
    this.poseApplied = false;

    // World-lock: bake current_eye · base_in_eye when a new registration
    // arrives, then hold it world-fixed between updates. This keeps head
    // micro-jitter out of the overlay (it doesn't vibrate against the stable
    // scene/passthrough) and is what makes it world-locked rather than
    // head-locked. No new poses (publishing stopped) ⇒ the pose just holds ⇒
    // auto lock-on-pause, for free.
    this._poseDirty = false;
    this._worldPose = new THREE.Matrix4();

    // Apply the Isaac extrinsic convention (ht_optical + CV->GL). Turn off to
    // treat the incoming 4x4 as a raw GL world transform.
    this.isaacConv = opts.isaacConv !== false;

    this.renderer = null;

    // ── Bag-replay stereo background (optional; enableBagBackground()) ──
    // Per-eye WebCodecs H.264 decoders -> canvas textures -> head-locked
    // quads drawn BENEATH the robot (depthTest off, renderOrder -10), sized
    // from the camera intrinsics so the recorded footage appears at the
    // correct FOV. Eye-selective visibility is handled in render().
    this._bag = null;
    this.refSpace = null;
    this._rt = null;
    this._urdfBaseURL = null;
    this._meshFiles = null;     // optional Map<basename, url> (drag-drop case)
  }

  // -- lifecycle ------------------------------------------------------------
  // Call from CloudXRClient's onWebGLInitialized(gl, _, referenceSpace).
  init(gl, referenceSpace) {
    this.refSpace = referenceSpace || null;
    this.renderer = new THREE.WebGLRenderer({
      canvas: gl.canvas, context: gl, antialias: false, alpha: true,
    });
    this.renderer.autoClear = false;        // never wipe the CloudXR frame
    this.renderer.autoClearColor = false;
    this.renderer.autoClearDepth = false;
    this.renderer.shadowMap.enabled = false;
    // Proxy render target we re-point at the XRWebGLLayer framebuffer.
    this._rt = new THREE.WebGLRenderTarget(1, 1);
  }

  setReferenceSpace(rs) { this.refSpace = rs; }

  // -- pose / joints (called by your transport) -----------------------------
  setPose(H) {                                  // H: THREE.Matrix4 (raw camera_to_base)
    let M;
    if (this.isaacConv) {
      const HO = H.clone().multiply(HT_OPTICAL); // H * ht_optical = cam-optical in base
      HO.invert();                               // base -> cam-optical (view)
      M = CV2GL.clone().multiply(HO);            // CV -> GL
    } else {
      M = H.clone();
    }
    this.poseRel.copy(M);
    this.poseApplied = true;
    this._poseDirty = true;
  }
  setPoseFromArray(a) {                          // 16 numbers, row-major
    this.setPose(new THREE.Matrix4().set(
      a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
      a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]));
  }
  setPoseFromPosQuat(p, q) {                     // {x,y,z}, {x,y,z,w}  (PoseStamped)
    this.setPose(new THREE.Matrix4().compose(
      new THREE.Vector3(p.x, p.y, p.z),
      new THREE.Quaternion(q.x, q.y, q.z, q.w),
      new THREE.Vector3(1, 1, 1)));
  }
  setJoints(positions, names) {
    if (!this.robot || !positions) return;
    const movable = Object.keys(this.robot.joints)
      .filter(n => this.robot.joints[n].jointType !== 'fixed');
    const byName = names && names.length === positions.length &&
                   names.some(n => this.robot.joints[n]);
    if (byName) {
      for (let i = 0; i < names.length; i++)
        if (this.robot.joints[names[i]]) this.robot.setJointValue(names[i], +positions[i]);
    } else {
      for (let i = 0; i < Math.min(positions.length, movable.length); i++)
        this.robot.setJointValue(movable[i], +positions[i]);
    }
  }

  // -- URDF loading (universal) ---------------------------------------------
  // Load ANY robot, from either source:
  //   loadURDF(url, opts)            – fetch a .urdf by URL
  //   loadURDFFromString(xml, opts)  – parse URDF XML you already hold,
  //                                    e.g. ROS /robot_description (which
  //                                    robot_state_publisher already expands
  //                                    from xacro, so no xacro step needed)
  // opts:
  //   packages   – package:// resolver for meshes; this is what makes arbitrary
  //                robots work. urdf-loader accepts a base-path string, a
  //                { packageName: baseURL } map, or a (pkg) => baseURL function.
  //   baseURL    – working path for relative mesh refs (default: the URDF URL
  //                for loadURDF; location.href for loadURDFFromString).
  //   meshFiles  – optional Map<basename, url> override (drag-drop case).
  //   parseCollision – default false (load visual meshes only).
  async loadURDF(url, opts = {}) {
    const baseURL = new URL(url, location.href).href;
    const text = await (await fetch(baseURL)).text();
    return this._parse(text, { ...opts, baseURL });
  }
  loadURDFFromString(xml, opts = {}) {
    const baseURL = opts.baseURL ? new URL(opts.baseURL, location.href).href : location.href;
    return this._parse(xml, { ...opts, baseURL });
  }
  _parse(text, opts) {
    this._urdfBaseURL = opts.baseURL;
    this._meshFiles = opts.meshFiles || null;
    const mgr = new THREE.LoadingManager();
    const loader = new URDFLoader(mgr);
    loader.packages = opts.packages ?? '';        // string base | {pkg:url} map | (pkg)=>url
    loader.parseCollision = opts.parseCollision ?? false;
    loader.workingPath = this._urdfBaseURL.slice(0, this._urdfBaseURL.lastIndexOf('/') + 1);
    loader.loadMeshCb = (path, manager, done) => this._loadMesh(path, manager, done);
    const robot = loader.parse(text);
    if (this.robot) this.poseGroup.remove(this.robot);
    this.robot = robot;
    this.poseGroup.add(robot);
    return robot;
  }

  _resolveMeshURL(path) {
    if (this._meshFiles) {                          // drag-drop override
      const base = path.split(/[\\/]/).pop();
      const stem = base.replace(/\.[^.]+$/, '');
      if (this._meshFiles.has(base)) return this._meshFiles.get(base);
      for (const [k, v] of this._meshFiles)         // match ignoring extension
        if (k.replace(/\.[^.]+$/, '') === stem) return v;
    }
    if (/^(https?:|data:|blob:)/.test(path)) return path;   // already resolved (packages map)
    const rel = path.replace(/^package:\/\/[^/]+\//, '');    // strip any unresolved package://
    return new URL(rel, this._urdfBaseURL).href;             // resolve against the URDF location
  }

  _loadMesh(path, manager, done) {
    const url = this._resolveMeshURL(path);
    const ext = url.split('?')[0].split('.').pop().toLowerCase();
    const fail = (e) => done(null, e || new Error('mesh load failed: ' + url));
    try {
      if (ext === 'stl') {
        new STLLoader(manager).load(url, (geo) => {
          const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial(
            { color: 0xcfd6e3, metalness: 0.1, roughness: 0.7 }));
          done(m);
        }, undefined, fail);
      } else if (ext === 'dae') {
        new ColladaLoader(manager).load(url, (c) => done(c.scene), undefined, fail);
      } else if (ext === 'obj') {
        new OBJLoader(manager).load(url, (o) => done(o), undefined, fail);
      } else if (ext === 'glb' || ext === 'gltf') {
        new GLTFLoader(manager).load(url, (g) => done(g.scene), undefined, fail);
      } else {
        fail(new Error('unsupported mesh extension: ' + ext));
      }
    } catch (e) { fail(e); }
  }

  // -- per-frame render -----------------------------------------------------
  // Call from onXRFrame(timestamp, frame, gl, baseLayer) AFTER CloudXR has
  // drawn the streamed frame. baseLayer.framebuffer is already bound by the
  // sample; we render into it without clearing color.
  // Build the replay background. camInfo: {fx, fy, width, height} (defaults
  // to the Quest passthrough calibration if omitted); depth: quad distance [m].
  enableBagBackground(camInfo = {}, depth = 2.0) {
    const fx = camInfo.fx ?? 864.75, fy = camInfo.fy ?? 864.75;
    const w = camInfo.width ?? 1280, h = camInfo.height ?? 1280;
    const qw = depth * w / fx, qh = depth * h / fy;
    const mk = (eye) => {
      const canvas = document.createElement('canvas');
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d');
      const tex = new THREE.CanvasTexture(canvas);
      tex.colorSpace = THREE.SRGBColorSpace;
      const mat = new THREE.MeshBasicMaterial({
        map: tex, depthTest: false, depthWrite: false, toneMapped: false });
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(qw, qh), mat);
      mesh.renderOrder = -10;
      mesh.frustumCulled = false;
      mesh.visible = false;
      mesh.matrixAutoUpdate = false;
      this.scene.add(mesh);
      let seq = 0;
      const dec = new VideoDecoder({
        output: (vf) => {
          ctx.drawImage(vf, 0, 0, w, h);
          vf.close();
          tex.needsUpdate = true;
        },
        error: (e) => console.warn(`[bag-bg] ${eye} decoder:`, e.message),
      });
      dec.configure({ codec: 'avc1.42E02A', optimizeForLatency: true });
      return { canvas, ctx, tex, mesh, dec, seq };
    };
    this._bag = { depth, left: mk('left'), right: mk('right'),
                  _m: new THREE.Matrix4(), _off: new THREE.Matrix4()
                      .makeTranslation(0, 0, -depth) };
    console.info(`[bag-bg] enabled: quad ${qw.toFixed(2)}x${qh.toFixed(2)} m `
                 + `at ${depth} m (fx=${fx}, ${w}x${h})`);
  }

  // Feed one H.264 access unit (Annex-B, all-intra) for one eye.
  setBagFrame(eye, u8) {
    if (!this._bag) return;
    const s = this._bag[eye];
    if (!s || s.dec.state !== 'configured') return;
    if (s.dec.decodeQueueSize > 2) return;          // latest-wins (all-intra)
    s.dec.decode(new EncodedVideoChunk(
      { type: 'key', timestamp: s.seq++ * 16667, data: u8 }));
  }

  render(frame, baseLayer) {
    if (!this.renderer || !this.refSpace || !this.poseApplied) return;
    const pose = frame.getViewerPose(this.refSpace);
    if (!pose || !pose.views.length) return;

    // Position the robot ONCE per frame using the head (left) eye, so both
    // eyes view one world-placed robot (correct stereo disparity).
    _eye.fromArray(pose.views[0].transform.matrix);

    // World-lock: bake current_eye · base_in_eye when a new registration
    // arrives, then HOLD it world-fixed between updates. Head motion no longer
    // re-enters frame-to-frame, so it stays planted on the target like the
    // scene/passthrough. No new poses (publishing stopped) ⇒ holds ⇒ auto-lock.
    if (this._poseDirty) {
      this._worldPose.multiplyMatrices(_eye, this.poseRel);
      this._poseDirty = false;
    }
    this.poseGroup.matrix.copy(this._worldPose);
    this.poseGroup.matrixWorldNeedsUpdate = true;

    // Hand three.js the same framebuffer CloudXR just drew into, via three's
    // externally-managed-framebuffer path (the same one its WebXRManager uses):
    // setting __useDefaultFramebuffer makes setRenderTarget bind our framebuffer
    // directly and skip three's own framebuffer/texture setup. (three r178+.)
    this.renderer.resetState();
    const props = this.renderer.properties.get(this._rt);
    props.__useDefaultFramebuffer = true;
    props.__webglFramebuffer = baseLayer.framebuffer;
    this._rt.width = baseLayer.framebufferWidth;
    this._rt.height = baseLayer.framebufferHeight;

    // Bind, then clear only depth (keep CloudXR's color) so the robot z-tests
    // internally and composites on top of the (depth-less) streamed video.
    this.renderer.setRenderTarget(this._rt);
    const gl = this.renderer.getContext();
    gl.clear(gl.DEPTH_BUFFER_BIT);

    for (const view of pose.views) {
      const vp = baseLayer.getViewport(view);
      this._rt.viewport.set(vp.x, vp.y, vp.width, vp.height);
      this._rt.scissor.set(vp.x, vp.y, vp.width, vp.height);
      this.renderer.setRenderTarget(this._rt);                       // rebind + this eye's viewport
      this.renderer.setViewport(vp.x, vp.y, vp.width, vp.height);     // also drives render()'s viewport
      this.camera.matrix.fromArray(view.transform.matrix);           // eye -> ref
      this.camera.matrixWorldNeedsUpdate = true;
      this.camera.projectionMatrix.fromArray(view.projectionMatrix);
      this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
      if (this._bag) {
        // place THIS eye's quad head-locked in front of THIS eye; show only it
        const isLeft = view.eye !== 'right';
        const cur = isLeft ? this._bag.left : this._bag.right;
        const oth = isLeft ? this._bag.right : this._bag.left;
        this._bag._m.fromArray(view.transform.matrix)
                    .multiply(this._bag._off);
        cur.mesh.matrix.copy(this._bag._m);
        cur.mesh.matrixWorldNeedsUpdate = true;
        cur.mesh.visible = true;
        oth.mesh.visible = false;
      }
      this.renderer.render(this.scene, this.camera);
    }
    this.renderer.setRenderTarget(null);
    // CloudXR re-binds its framebuffer + state at the top of the next frame,
    // so no explicit GL-state restore is needed here.
  }
}