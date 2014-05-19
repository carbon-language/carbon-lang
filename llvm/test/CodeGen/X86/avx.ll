; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s

define <4 x i32> @blendvb_fallback_v4i32(<4 x i1> %mask, <4 x i32> %x, <4 x i32> %y) {
; CHECK-LABEL: @blendvb_fallback_v4i32
; CHECK: vblendvps
; CHECK: ret
  %ret = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %ret
}

define <8 x i32> @blendvb_fallback_v8i32(<8 x i1> %mask, <8 x i32> %x, <8 x i32> %y) {
; CHECK-LABEL: @blendvb_fallback_v8i32
; CHECK: vblendvps
; CHECK: ret
  %ret = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %y
  ret <8 x i32> %ret
}

define <8 x float> @blendvb_fallback_v8f32(<8 x i1> %mask, <8 x float> %x, <8 x float> %y) {
; CHECK-LABEL: @blendvb_fallback_v8f32
; CHECK: vblendvps
; CHECK: ret
  %ret = select <8 x i1> %mask, <8 x float> %x, <8 x float> %y
  ret <8 x float> %ret
}
