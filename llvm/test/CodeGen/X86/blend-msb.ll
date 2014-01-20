; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 -mattr=+sse4.1 | FileCheck %s


; Verify that we produce movss instead of blendvps when possible.

;CHECK-LABEL: vsel_float:
;CHECK-NOT: blendvps
;CHECK: movss
;CHECK: ret
define <4 x float> @vsel_float(<4 x float> %v1, <4 x float> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}

;CHECK-LABEL: vsel_4xi8:
;CHECK-NOT: blendvps
;CHECK: movss
;CHECK: ret
define <4 x i8> @vsel_4xi8(<4 x i8> %v1, <4 x i8> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i8> %v1, <4 x i8> %v2
  ret <4 x i8> %vsel
}


; We do not have native support for v8i16 blends and we have to use the
; blendvb instruction or a sequence of NAND/OR/AND. Make sure that we do not
; reduce the mask in this case.
;CHECK-LABEL: vsel_8xi16:
;CHECK: andps
;CHECK: andps
;CHECK: orps
;CHECK: ret
define <8 x i16> @vsel_8xi16(<8 x i16> %v1, <8 x i16> %v2) {
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x i16> %v1, <8 x i16> %v2
  ret <8 x i16> %vsel
}
