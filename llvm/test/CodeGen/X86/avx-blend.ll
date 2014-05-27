; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx  -mattr=+avx | FileCheck %s

; AVX128 tests:

;CHECK-LABEL: vsel_float:
; select mask is <i1 true, i1 false, i1 true, i1 false>.
; Big endian representation is 0101 = 5.
; '1' means takes the first argument, '0' means takes the second argument.
; This is the opposite of the intel syntax, thus we expect
; the inverted mask: 1010 = 10.
; According to the ABI:
; v1 is in xmm0 => first argument is xmm0.
; v2 is in xmm1 => second argument is xmm1.
; result is in xmm0 => destination argument.
;CHECK: vblendps    $10, %xmm1, %xmm0, %xmm0
;CHECK: ret
define <4 x float> @vsel_float(<4 x float> %v1, <4 x float> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}


;CHECK-LABEL: vsel_i32:
;CHECK: vblendps   $10, %xmm1, %xmm0, %xmm0
;CHECK: ret
define <4 x i32> @vsel_i32(<4 x i32> %v1, <4 x i32> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i32> %v1, <4 x i32> %v2
  ret <4 x i32> %vsel
}


;CHECK-LABEL: vsel_double:
;CHECK: vmovsd
;CHECK: ret
define <2 x double> @vsel_double(<2 x double> %v1, <2 x double> %v2) {
  %vsel = select <2 x i1> <i1 true, i1 false>, <2 x double> %v1, <2 x double> %v2
  ret <2 x double> %vsel
}


;CHECK-LABEL: vsel_i64:
;CHECK: vmovsd
;CHECK: ret
define <2 x i64> @vsel_i64(<2 x i64> %v1, <2 x i64> %v2) {
  %vsel = select <2 x i1> <i1 true, i1 false>, <2 x i64> %v1, <2 x i64> %v2
  ret <2 x i64> %vsel
}


;CHECK-LABEL: vsel_i8:
;CHECK: vpblendvb
;CHECK: ret
define <16 x i8> @vsel_i8(<16 x i8> %v1, <16 x i8> %v2) {
  %vsel = select <16 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <16 x i8> %v1, <16 x i8> %v2
  ret <16 x i8> %vsel
}


; AVX256 tests:


;CHECK-LABEL: vsel_float8:
;CHECK-NOT: vinsertf128
; <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>
; which translates into the boolean mask (big endian representation):
; 00010001 = 17.
; '1' means takes the first argument, '0' means takes the second argument.
; This is the opposite of the intel syntax, thus we expect
; the inverted mask: 11101110 = 238.
;CHECK: vblendps    $238, %ymm1, %ymm0, %ymm0
;CHECK: ret
define <8 x float> @vsel_float8(<8 x float> %v1, <8 x float> %v2) {
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x float> %v1, <8 x float> %v2
  ret <8 x float> %vsel
}

;CHECK-LABEL: vsel_i328:
;CHECK-NOT: vinsertf128
;CHECK: vblendps    $238, %ymm1, %ymm0, %ymm0
;CHECK-NEXT: ret
define <8 x i32> @vsel_i328(<8 x i32> %v1, <8 x i32> %v2) {
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x i32> %v1, <8 x i32> %v2
  ret <8 x i32> %vsel
}

;CHECK-LABEL: vsel_double8:
; select mask is 2x: 0001 => intel mask: ~0001 = 14
; ABI:
; v1 is in ymm0 and ymm1.
; v2 is in ymm2 and ymm3.
; result is in ymm0 and ymm1.
; Compute the low part: res.low = blend v1.low, v2.low, blendmask
;CHECK: vblendpd    $14, %ymm2, %ymm0, %ymm0
; Compute the high part.
;CHECK: vblendpd    $14, %ymm3, %ymm1, %ymm1
;CHECK: ret
define <8 x double> @vsel_double8(<8 x double> %v1, <8 x double> %v2) {
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x double> %v1, <8 x double> %v2
  ret <8 x double> %vsel
}

;CHECK-LABEL: vsel_i648:
;CHECK: vblendpd    $14, %ymm2, %ymm0, %ymm0
;CHECK: vblendpd    $14, %ymm3, %ymm1, %ymm1
;CHECK: ret
define <8 x i64> @vsel_i648(<8 x i64> %v1, <8 x i64> %v2) {
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x i64> %v1, <8 x i64> %v2
  ret <8 x i64> %vsel
}

;CHECK-LABEL: vsel_double4:
;CHECK-NOT: vinsertf128
;CHECK: vshufpd $10
;CHECK-NEXT: ret
define <4 x double> @vsel_double4(<4 x double> %v1, <4 x double> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x double> %v1, <4 x double> %v2
  ret <4 x double> %vsel
}

;; TEST blend + compares
; CHECK: testa
define <2 x double> @testa(<2 x double> %x, <2 x double> %y) {
  ; CHECK: vcmplepd
  ; CHECK: vblendvpd
  %max_is_x = fcmp oge <2 x double> %x, %y
  %max = select <2 x i1> %max_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %max
}

; CHECK: testb
define <2 x double> @testb(<2 x double> %x, <2 x double> %y) {
  ; CHECK: vcmpnlepd
  ; CHECK: vblendvpd
  %min_is_x = fcmp ult <2 x double> %x, %y
  %min = select <2 x i1> %min_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %min
}

; If we can figure out a blend has a constant mask, we should emit the
; blend instruction with an immediate mask
define <4 x double> @constant_blendvpd_avx(<4 x double> %xy, <4 x double> %ab) {
; CHECK-LABEL: constant_blendvpd_avx:
; CHECK-NOT: mov
; CHECK: vblendpd
; CHECK: ret
  %1 = select <4 x i1> <i1 false, i1 false, i1 true, i1 false>, <4 x double> %xy, <4 x double> %ab
  ret <4 x double> %1
}

define <8 x float> @constant_blendvps_avx(<8 x float> %xyzw, <8 x float> %abcd) {
; CHECK-LABEL: constant_blendvps_avx:
; CHECK-NOT: mov
; CHECK: vblendps
; CHECK: ret
  %1 = select <8 x i1> <i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true>, <8 x float> %xyzw, <8 x float> %abcd
  ret <8 x float> %1
}

declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>)
declare <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double>, <4 x double>, <4 x double>)
