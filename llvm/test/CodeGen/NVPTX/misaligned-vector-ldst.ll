; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: t1
define <4 x float> @t1(i8* %p1) {
; CHECK-NOT: ld.v4
; CHECK-NOT: ld.v2
; CHECK-NOT: ld.f32
; CHECK: ld.u8
  %cast = bitcast i8* %p1 to <4 x float>*
  %r = load <4 x float>* %cast, align 1
  ret <4 x float> %r
}

; CHECK-LABEL: t2
define <4 x float> @t2(i8* %p1) {
; CHECK-NOT: ld.v4
; CHECK-NOT: ld.v2
; CHECK: ld.f32
  %cast = bitcast i8* %p1 to <4 x float>*
  %r = load <4 x float>* %cast, align 4
  ret <4 x float> %r
}

; CHECK-LABEL: t3
define <4 x float> @t3(i8* %p1) {
; CHECK-NOT: ld.v4
; CHECK: ld.v2
  %cast = bitcast i8* %p1 to <4 x float>*
  %r = load <4 x float>* %cast, align 8
  ret <4 x float> %r
}

; CHECK-LABEL: t4
define <4 x float> @t4(i8* %p1) {
; CHECK: ld.v4
  %cast = bitcast i8* %p1 to <4 x float>*
  %r = load <4 x float>* %cast, align 16
  ret <4 x float> %r
}


; CHECK-LABEL: s1
define void @s1(<4 x float>* %p1, <4 x float> %v) {
; CHECK-NOT: st.v4
; CHECK-NOT: st.v2
; CHECK-NOT: st.f32
; CHECK: st.u8
  store <4 x float> %v, <4 x float>* %p1, align 1
  ret void
}

; CHECK-LABEL: s2
define void @s2(<4 x float>* %p1, <4 x float> %v) {
; CHECK-NOT: st.v4
; CHECK-NOT: st.v2
; CHECK: st.f32
  store <4 x float> %v, <4 x float>* %p1, align 4
  ret void
}

; CHECK-LABEL: s3
define void @s3(<4 x float>* %p1, <4 x float> %v) {
; CHECK-NOT: st.v4
  store <4 x float> %v, <4 x float>* %p1, align 8
  ret void
}

; CHECK-LABEL: s4
define void @s4(<4 x float>* %p1, <4 x float> %v) {
; CHECK: st.v4
  store <4 x float> %v, <4 x float>* %p1, align 16
  ret void
}

