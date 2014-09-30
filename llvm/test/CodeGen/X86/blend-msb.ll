; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 -mattr=+sse4.1 | FileCheck %s


; Verify that we produce movss instead of blendvps when possible.

;CHECK-LABEL: vsel_float:
;CHECK-NOT: blend
;CHECK: movss
;CHECK: ret
define <4 x float> @vsel_float(<4 x float> %v1, <4 x float> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}

;CHECK-LABEL: vsel_4xi8:
;CHECK-NOT: blend
;CHECK: movss
;CHECK: ret
define <4 x i8> @vsel_4xi8(<4 x i8> %v1, <4 x i8> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i8> %v1, <4 x i8> %v2
  ret <4 x i8> %vsel
}

;CHECK-LABEL: vsel_8xi16:
; The select mask is
; <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>
; which translates into the boolean mask (big endian representation):
; 00010001 = 17.
; '1' means takes the first argument, '0' means takes the second argument.
; This is the opposite of the intel syntax, thus we expect
; the inverted mask: 11101110 = 238.
; According to the ABI:
; v1 is in xmm0 => first argument is xmm0.
; v2 is in xmm1 => second argument is xmm1.
;CHECK: pblendw $238, %xmm1, %xmm0
;CHECK: ret
define <8 x i16> @vsel_8xi16(<8 x i16> %v1, <8 x i16> %v2) {
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x i16> %v1, <8 x i16> %v2
  ret <8 x i16> %vsel
}
