import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

/** Shared depth/CoC helpers for both stages. */
const DOF_GLSL_COMMON = /* glsl */ `
  #include <common>
  #include <packing>
  varying vec2 vUv;
  uniform sampler2D tDepth;
  uniform float focus;
  uniform float focalDepth;
  uniform float aperture;
  uniform float maxblur;
  uniform float nearClip;
  uniform float farClip;

  float viewDist( const in vec2 uv ) {
    float d = unpackRGBAToDepth( texture2D( tDepth, uv ) );
    return -perspectiveDepthToViewZ( d, nearClip, farClip );
  }

  float cocPx( const in float dist, const in float pxY ) {
    float excess = abs( dist - focus ) - focalDepth;
    if ( excess <= 0.0 ) return 0.0;
    return min( aperture * excess / max( dist, 0.2 ), maxblur ) / pxY;
  }
`;

const GATHER_FRAGMENT = /* glsl */ `
  ${''}
  uniform sampler2D tColor;

  void main() {
    vec2 px = vec2( abs( dFdx( vUv.x ) ), abs( dFdy( vUv.y ) ) );
    float pxY = max( px.y, 1e-6 );
    float maxRadPx = maxblur / pxY;

    vec4 center = texture2D( tColor, vUv );
    float centerDist = viewDist( vUv );
    float centerCoc = cocPx( centerDist, pxY );

    // adaptive reach: full radius only where the pixel (or potential
    // foreground spill) needs it
    float radius = min( maxRadPx, max( centerCoc * 1.15, maxRadPx * 0.25 ) );

    // flatness early-out: fully-blurred flat regions (mostly the solid
    // background) blur to themselves — probe a ring and skip the gather
    if ( centerCoc >= maxRadPx * 0.95 ) {
      vec4 p0 = texture2D( tColor, vUv + vec2( radius * px.x, 0.0 ) );
      vec4 p1 = texture2D( tColor, vUv - vec2( radius * px.x, 0.0 ) );
      vec4 p2 = texture2D( tColor, vUv + vec2( 0.0, radius * px.y ) );
      vec4 p3 = texture2D( tColor, vUv - vec2( 0.0, radius * px.y ) );
      vec4 avg = ( p0 + p1 + p2 + p3 ) * 0.25;
      float dev = dot( abs( p0 - avg ).rgb + abs( p1 - avg ).rgb + abs( p2 - avg ).rgb + abs( p3 - avg ).rgb, vec3( 1.0 ) )
        + abs( center.a - avg.a ) * 4.0;
      if ( dev < 0.02 ) {
        gl_FragColor = mix( center, avg, 0.5 );
        return;
      }
    }

    // per-pixel rotation turns ring artifacts into fine noise
    float rnd = fract( sin( dot( gl_FragCoord.xy, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 ) * 6.28318;

    vec4 acc = center;
    float tot = 1.0;
    const int N = 96;
    // tap count scales with the disc area actually gathered
    float nEff = clamp( radius * radius * 0.1, 12.0, float( N ) );
    const float GOLDEN = 2.39996323;
    for ( int i = 1; i <= N; i ++ ) {
      float fi = float( i );
      if ( fi > nEff ) break;
      float ang = fi * GOLDEN + rnd;
      float rad = radius * sqrt( fi / nEff );
      vec2 tc = vUv + vec2( cos( ang ) * px.x, sin( ang ) * px.y ) * rad;
      vec4 sColor = texture2D( tColor, tc );
      float sDist = viewDist( tc );
      float sCoc = cocPx( sDist, pxY );
      // background may not bleed over a sharp foreground
      if ( sDist > centerDist ) sCoc = min( sCoc, centerCoc * 2.0 + 1.0 );
      float m = smoothstep( rad - 1.0, rad + 1.0, sCoc );
      acc += mix( acc / tot, sColor, m );
      tot += 1.0;
    }
    gl_FragColor = acc / tot;
  }
`;

const COMPOSITE_FRAGMENT = /* glsl */ `
  uniform sampler2D tSharp;
  uniform sampler2D tBlur;

  void main() {
    vec4 sharp = texture2D( tSharp, vUv );
    vec4 blur = texture2D( tBlur, vUv );
    float pxY = max( abs( dFdy( vUv.y ) ), 1e-6 );
    float coc = cocPx( viewDist( vUv ), pxY );
    float w = smoothstep( 0.3, 1.5, coc );
    gl_FragColor = mix( sharp, blur, w );
  }
`;

const DOF_VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
  }
`;

function makeDofUniforms() {
  return {
    tDepth: { value: null as THREE.Texture | null },
    focus: { value: 5.2 },
    focalDepth: { value: 0.15 },
    aperture: { value: 0.4 },
    maxblur: { value: 0.04 },
    nearClip: { value: 0.1 },
    farClip: { value: 200 },
  };
}

/**
 * Macro depth of field: depth pre-pass, half-resolution adaptive
 * circle-of-confusion gather (golden spiral, per-tap depth gating),
 * full-resolution composite. Half-res + adaptive tap count keep large blur
 * radii cheap; blurred regions cannot resolve detail, so the downsample is
 * invisible in practice.
 */
export class MacroDofPass extends Pass {
  private sceneRef: THREE.Scene;
  private cameraRef: THREE.PerspectiveCamera;
  private depthMaterial: THREE.MeshDepthMaterial;
  private depthRT: THREE.WebGLRenderTarget;
  private blurRT: THREE.WebGLRenderTarget;
  private gatherMat: THREE.ShaderMaterial;
  private compositeMat: THREE.ShaderMaterial;
  private fsQuad: FullScreenQuad;
  private clearColorTmp = new THREE.Color();

  constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    super();
    this.sceneRef = scene;
    this.cameraRef = camera;
    this.needsSwap = true;

    this.depthMaterial = new THREE.MeshDepthMaterial();
    this.depthMaterial.depthPacking = THREE.RGBADepthPacking;
    this.depthMaterial.blending = THREE.NoBlending;

    this.depthRT = new THREE.WebGLRenderTarget(1, 1, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
    });
    this.blurRT = new THREE.WebGLRenderTarget(1, 1, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      type: THREE.HalfFloatType,
    });

    this.gatherMat = new THREE.ShaderMaterial({
      uniforms: { ...makeDofUniforms(), tColor: { value: null } },
      vertexShader: DOF_VERTEX,
      fragmentShader: DOF_GLSL_COMMON + GATHER_FRAGMENT,
    });
    this.compositeMat = new THREE.ShaderMaterial({
      uniforms: { ...makeDofUniforms(), tSharp: { value: null }, tBlur: { value: null } },
      vertexShader: DOF_VERTEX,
      fragmentShader: DOF_GLSL_COMMON + COMPOSITE_FRAGMENT,
    });

    this.fsQuad = new FullScreenQuad(this.gatherMat);
  }

  /** Push a value into the matching uniform of both stages. */
  private setBoth(name: string, value: number) {
    (this.gatherMat.uniforms as Record<string, THREE.IUniform>)[name].value = value;
    (this.compositeMat.uniforms as Record<string, THREE.IUniform>)[name].value = value;
  }

  setParams(aperture: number, maxblur: number, focalDepth: number) {
    this.setBoth('aperture', aperture);
    this.setBoth('maxblur', maxblur);
    this.setBoth('focalDepth', focalDepth);
  }

  setFocus(distance: number) {
    this.setBoth('focus', distance);
  }

  override setSize(width: number, height: number) {
    this.depthRT.setSize(width, height);
    this.blurRT.setSize(Math.max(1, width >> 1), Math.max(1, height >> 1));
  }

  override render(
    renderer: THREE.WebGLRenderer,
    writeBuffer: THREE.WebGLRenderTarget,
    readBuffer: THREE.WebGLRenderTarget,
  ) {
    this.setBoth('nearClip', this.cameraRef.near);
    this.setBoth('farClip', this.cameraRef.far);

    // 1. scene depth (background clears to far)
    renderer.getClearColor(this.clearColorTmp);
    const oldAlpha = renderer.getClearAlpha();
    const oldAutoClear = renderer.autoClear;
    renderer.autoClear = false;
    this.sceneRef.overrideMaterial = this.depthMaterial;
    renderer.setClearColor(0xffffff, 1);
    renderer.setRenderTarget(this.depthRT);
    renderer.clear();
    renderer.render(this.sceneRef, this.cameraRef);
    this.sceneRef.overrideMaterial = null;
    renderer.setClearColor(this.clearColorTmp, oldAlpha);

    // 2. half-resolution adaptive gather
    this.gatherMat.uniforms.tColor.value = readBuffer.texture;
    this.gatherMat.uniforms.tDepth.value = this.depthRT.texture;
    this.fsQuad.material = this.gatherMat;
    renderer.setRenderTarget(this.blurRT);
    this.fsQuad.render(renderer);

    // 3. full-resolution composite
    this.compositeMat.uniforms.tSharp.value = readBuffer.texture;
    this.compositeMat.uniforms.tBlur.value = this.blurRT.texture;
    this.compositeMat.uniforms.tDepth.value = this.depthRT.texture;
    this.fsQuad.material = this.compositeMat;
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    this.fsQuad.render(renderer);

    renderer.autoClear = oldAutoClear;
  }

  override dispose() {
    this.depthMaterial.dispose();
    this.depthRT.dispose();
    this.blurRT.dispose();
    this.gatherMat.dispose();
    this.compositeMat.dispose();
    this.fsQuad.dispose();
  }
}
