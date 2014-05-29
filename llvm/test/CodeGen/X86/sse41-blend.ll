; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 -mattr=+sse4.1 | FileCheck %s

;CHECK-LABEL: vsel_float:
;CHECK: blendps
;CHECK: ret
define <4 x float> @vsel_float(<4 x float> %v1, <4 x float> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}


;CHECK-LABEL: vsel_4xi8:
;CHECK: blendps
;CHECK: ret
define <4 x i8> @vsel_4xi8(<4 x i8> %v1, <4 x i8> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 true, i1 false, i1 true>, <4 x i8> %v1, <4 x i8> %v2
  ret <4 x i8> %vsel
}

;CHECK-LABEL: vsel_4xi16:
;CHECK: blendps
;CHECK: ret
define <4 x i16> @vsel_4xi16(<4 x i16> %v1, <4 x i16> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x i16> %v1, <4 x i16> %v2
  ret <4 x i16> %vsel
}


;CHECK-LABEL: vsel_i32:
;CHECK: blendps
;CHECK: ret
define <4 x i32> @vsel_i32(<4 x i32> %v1, <4 x i32> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 true, i1 false, i1 true>, <4 x i32> %v1, <4 x i32> %v2
  ret <4 x i32> %vsel
}


;CHECK-LABEL: vsel_double:
;CHECK: movsd
;CHECK: ret
define <4 x double> @vsel_double(<4 x double> %v1, <4 x double> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x double> %v1, <4 x double> %v2
  ret <4 x double> %vsel
}


;CHECK-LABEL: vsel_i64:
;CHECK: movsd
;CHECK: ret
define <4 x i64> @vsel_i64(<4 x i64> %v1, <4 x i64> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i64> %v1, <4 x i64> %v2
  ret <4 x i64> %vsel
}


;CHECK-LABEL: vsel_i8:
;CHECK: pblendvb
;CHECK: ret
define <16 x i8> @vsel_i8(<16 x i8> %v1, <16 x i8> %v2) {
  %vsel = select <16 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <16 x i8> %v1, <16 x i8> %v2
  ret <16 x i8> %vsel
}

;; TEST blend + compares
; CHECK: A
define <2 x double> @A(<2 x double> %x, <2 x double> %y) {
  ; CHECK: cmplepd
  ; CHECK: blendvpd
  %max_is_x = fcmp oge <2 x double> %x, %y
  %max = select <2 x i1> %max_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %max
}

; CHECK: B
define <2 x double> @B(<2 x double> %x, <2 x double> %y) {
  ; CHECK: cmpnlepd
  ; CHECK: blendvpd
  %min_is_x = fcmp ult <2 x double> %x, %y
  %min = select <2 x i1> %min_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %min
}

; CHECK: float_crash
define void @float_crash() nounwind {
entry:
  %merge205vector_func.i = select <4 x i1> undef, <4 x double> undef, <4 x double> undef
  %extract214vector_func.i = extractelement <4 x double> %merge205vector_func.i, i32 0
  store double %extract214vector_func.i, double addrspace(1)* undef, align 8
  ret void
}

; If we can figure out a blend has a constant mask, we should emit the
; blend instruction with an immediate mask
define <2 x double> @constant_blendvpd(<2 x double> %xy, <2 x double> %ab) {
; In this case, we emit a simple movss
; CHECK-LABEL: constant_blendvpd
; CHECK: movsd
; CHECK: ret
  %1 = select <2 x i1> <i1 true, i1 false>, <2 x double> %xy, <2 x double> %ab
  ret <2 x double> %1
}

define <4 x float> @constant_blendvps(<4 x float> %xyzw, <4 x float> %abcd) {
; CHECK-LABEL: constant_blendvps
; CHECK-NOT: mov
; CHECK: blendps $7
; CHECK: ret
  %1 = select <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x float> %xyzw, <4 x float> %abcd
  ret <4 x float> %1
}

define <16 x i8> @constant_pblendvb(<16 x i8> %xyzw, <16 x i8> %abcd) {
; CHECK-LABEL: constant_pblendvb:
; CHECK: movaps
; CHECK: pblendvb
; CHECK: ret
  %1 = select <16 x i1> <i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false>, <16 x i8> %xyzw, <16 x i8> %abcd
  ret <16 x i8> %1
}

declare <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8>, <16 x i8>, <16 x i8>)
declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>)
declare <2 x double> @llvm.x86.sse41.blendvpd(<2 x double>, <2 x double>, <2 x double>)

;; 2 tests for shufflevectors that optimize to blend + immediate
; CHECK-LABEL: @blend_shufflevector_4xfloat
; CHECK: blendps
; CHECK: ret
define <4 x float> @blend_shufflevector_4xfloat(<4 x float> %a, <4 x float> %b) {
  %1 = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 3>
  ret <4 x float> %1
}

; CHECK-LABEL: @blend_shufflevector_8xi16
; CHECK: pblendw
; CHECK: ret
define <8 x i16> @blend_shufflevector_8xi16(<8 x i16> %a, <8 x i16> %b) {
  %1 = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 3, i32 4, i32 5, i32 6, i32 15>
  ret <8 x i16> %1
}
