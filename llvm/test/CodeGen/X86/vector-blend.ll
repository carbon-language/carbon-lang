; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx  -mattr=+avx | FileCheck %s

; AVX128 tests:

define <4 x float> @vsel_float(<4 x float> %v1, <4 x float> %v2) {
; CHECK-LABEL: vsel_float:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendps {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; CHECK-NEXT:    retq
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}

define <4 x float> @vsel_float2(<4 x float> %v1, <4 x float> %v2) {
; CHECK-LABEL: vsel_float2:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vmovss %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}

define <4 x i32> @vsel_i32(<4 x i32> %v1, <4 x i32> %v2) {
; CHECK-LABEL: vsel_i32:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendps {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; CHECK-NEXT:    retq
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i32> %v1, <4 x i32> %v2
  ret <4 x i32> %vsel
}

define <2 x double> @vsel_double(<2 x double> %v1, <2 x double> %v2) {
; CHECK-LABEL: vsel_double:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vmovsd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %vsel = select <2 x i1> <i1 true, i1 false>, <2 x double> %v1, <2 x double> %v2
  ret <2 x double> %vsel
}

define <2 x i64> @vsel_i64(<2 x i64> %v1, <2 x i64> %v2) {
; CHECK-LABEL: vsel_i64:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vmovsd %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %vsel = select <2 x i1> <i1 true, i1 false>, <2 x i64> %v1, <2 x i64> %v2
  ret <2 x i64> %vsel
}

define <8 x i16> @vsel_8xi16(<8 x i16> %v1, <8 x i16> %v2) {
; CHECK-LABEL: vsel_8xi16:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],xmm1[1,2,3],xmm0[4],xmm1[5,6,7]
; CHECK-NEXT:    retq
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x i16> %v1, <8 x i16> %v2
  ret <8 x i16> %vsel
}

define <16 x i8> @vsel_i8(<16 x i8> %v1, <16 x i8> %v2) {
; CHECK-LABEL: vsel_i8:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vmovdqa {{.*#+}} xmm2 = [255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0]
; CHECK-NEXT:    vpblendvb %xmm2, %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %vsel = select <16 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <16 x i8> %v1, <16 x i8> %v2
  ret <16 x i8> %vsel
}


; AVX256 tests:

define <8 x float> @vsel_float8(<8 x float> %v1, <8 x float> %v2) {
; CHECK-LABEL: vsel_float8:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3],ymm0[4],ymm1[5,6,7]
; CHECK-NEXT:    retq
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x float> %v1, <8 x float> %v2
  ret <8 x float> %vsel
}

define <8 x i32> @vsel_i328(<8 x i32> %v1, <8 x i32> %v2) {
; CHECK-LABEL: vsel_i328:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3],ymm0[4],ymm1[5,6,7]
; CHECK-NEXT:    retq
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x i32> %v1, <8 x i32> %v2
  ret <8 x i32> %vsel
}

define <8 x double> @vsel_double8(<8 x double> %v1, <8 x double> %v2) {
; CHECK-LABEL: vsel_double8:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm2[1,2,3]
; CHECK-NEXT:    vblendpd {{.*#+}} ymm1 = ymm1[0],ymm3[1,2,3]
; CHECK-NEXT:    retq
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x double> %v1, <8 x double> %v2
  ret <8 x double> %vsel
}

define <8 x i64> @vsel_i648(<8 x i64> %v1, <8 x i64> %v2) {
; CHECK-LABEL: vsel_i648:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm2[1,2,3]
; CHECK-NEXT:    vblendpd {{.*#+}} ymm1 = ymm1[0],ymm3[1,2,3]
; CHECK-NEXT:    retq
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x i64> %v1, <8 x i64> %v2
  ret <8 x i64> %vsel
}

define <4 x double> @vsel_double4(<4 x double> %v1, <4 x double> %v2) {
; CHECK-LABEL: vsel_double4:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3]
; CHECK-NEXT:    retq
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x double> %v1, <4 x double> %v2
  ret <4 x double> %vsel
}

define <2 x double> @testa(<2 x double> %x, <2 x double> %y) {
; CHECK-LABEL: testa:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcmplepd %xmm0, %xmm1, %xmm2
; CHECK-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %max_is_x = fcmp oge <2 x double> %x, %y
  %max = select <2 x i1> %max_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %max
}

define <2 x double> @testb(<2 x double> %x, <2 x double> %y) {
; CHECK-LABEL: testb:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcmpnlepd %xmm0, %xmm1, %xmm2
; CHECK-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %min_is_x = fcmp ult <2 x double> %x, %y
  %min = select <2 x i1> %min_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %min
}

; If we can figure out a blend has a constant mask, we should emit the
; blend instruction with an immediate mask
define <4 x double> @constant_blendvpd_avx(<4 x double> %xy, <4 x double> %ab) {
; CHECK-LABEL: constant_blendvpd_avx:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendpd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2],ymm1[3]
; CHECK-NEXT:    retq
  %1 = select <4 x i1> <i1 false, i1 false, i1 true, i1 false>, <4 x double> %xy, <4 x double> %ab
  ret <4 x double> %1
}

define <8 x float> @constant_blendvps_avx(<8 x float> %xyzw, <8 x float> %abcd) {
; CHECK-LABEL: constant_blendvps_avx:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendps {{.*#+}} ymm0 = ymm1[0,1,2],ymm0[3],ymm1[4,5,6],ymm0[7]
; CHECK-NEXT:    retq
  %1 = select <8 x i1> <i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true>, <8 x float> %xyzw, <8 x float> %abcd
  ret <8 x float> %1
}

declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>)
declare <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double>, <4 x double>, <4 x double>)

;; 4 tests for shufflevectors that optimize to blend + immediate
define <4 x float> @blend_shufflevector_4xfloat(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: blend_shufflevector_4xfloat:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendps {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; CHECK-NEXT:    retq
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x float> %1
}

define <8 x float> @blend_shufflevector_8xfloat(<8 x float> %a, <8 x float> %b) {
; CHECK-LABEL: blend_shufflevector_8xfloat:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3,4,5],ymm0[6],ymm1[7]
; CHECK-NEXT:    retq
  %1 = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 12, i32 13, i32 6, i32 15>
  ret <8 x float> %1
}

define <4 x double> @blend_shufflevector_4xdouble(<4 x double> %a, <4 x double> %b) {
; CHECK-LABEL: blend_shufflevector_4xdouble:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2,3]
; CHECK-NEXT:    retq
  %1 = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 3>
  ret <4 x double> %1
}

define <4 x i64> @blend_shufflevector_4xi64(<4 x i64> %a, <4 x i64> %b) {
; CHECK-LABEL: blend_shufflevector_4xi64:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vblendpd {{.*#+}} ymm0 = ymm1[0],ymm0[1],ymm1[2,3]
; CHECK-NEXT:    retq
  %1 = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  ret <4 x i64> %1
}
