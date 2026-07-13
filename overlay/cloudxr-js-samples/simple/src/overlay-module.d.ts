// overlay-module.d.ts — types for overlay-module.js (bundler resolution finds
// this for `import { OverlayModule } from './overlay-module.js'`; webpack still
// bundles the real .js at runtime).

import * as THREE from 'three';

export interface OverlayOptions {
  isaacConv?: boolean;   // default true: apply ht_optical + CV->GL convention
}

export interface LoadURDFOptions {
  // package:// resolver for meshes: base-path string, { pkg: baseURL } map,
  // or (pkg) => baseURL function.
  packages?: string | Record<string, string> | ((pkg: string) => string);
  baseURL?: string;
  meshFiles?: Map<string, string>;
  parseCollision?: boolean;
}

export declare class OverlayModule {
  constructor(opts?: OverlayOptions);

  // lifecycle
  init(gl: WebGL2RenderingContext, referenceSpace: XRReferenceSpace): void;
  setReferenceSpace(rs: XRReferenceSpace): void;

  // pose / joints (call from your transport: data channel, rosbridge, ws, …)
  setPose(H: THREE.Matrix4): void;
  setPoseFromArray(a: ArrayLike<number>): void;          // 16, row-major = camera_to_base
  setPoseFromPosQuat(
    p: { x: number; y: number; z: number },
    q: { x: number; y: number; z: number; w: number }
  ): void;
  setJoints(positions: ArrayLike<number>, names?: string[]): void;
  enableBagBackground(camInfo?: { fx?: number; fy?: number;
                                  width?: number; height?: number },
                      depth?: number): void;
  setBagFrame(eye: 'left' | 'right', data: Uint8Array): void;

  // robot loading
  loadURDF(url: string, opts?: LoadURDFOptions): Promise<unknown>;
  loadURDFFromString(xml: string, opts?: LoadURDFOptions): unknown;

  // per-frame render (call from onXRFrame, after CloudXR draws). World-locked:
  // bakes camera_to_base into a world pose on each update and holds it.
  render(frame: XRFrame, baseLayer: XRWebGLLayer): void;
}