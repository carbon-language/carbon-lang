; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128-n32"

define void @foo1(i16* %p, i16* %r) nounwind {
entry:
  %v = load i16* %p, align 1
  store i16 %v, i16* %r, align 1
  ret void

; CHECK: @foo1
; CHECK: lhz
; CHECK: sth
}

define void @foo2(i32* %p, i32* %r) nounwind {
entry:
  %v = load i32* %p, align 1
  store i32 %v, i32* %r, align 1
  ret void

; CHECK: @foo2
; CHECK: lwz
; CHECK: stw
}

define void @foo3(i64* %p, i64* %r) nounwind {
entry:
  %v = load i64* %p, align 1
  store i64 %v, i64* %r, align 1
  ret void

; CHECK: @foo3
; CHECK: ld
; CHECK: std
}

define void @foo4(float* %p, float* %r) nounwind {
entry:
  %v = load float* %p, align 1
  store float %v, float* %r, align 1
  ret void

; CHECK: @foo4
; CHECK: lfs
; CHECK: stfs
}

define void @foo5(double* %p, double* %r) nounwind {
entry:
  %v = load double* %p, align 1
  store double %v, double* %r, align 1
  ret void

; CHECK: @foo5
; CHECK: lfd
; CHECK: stfd
}

define void @foo6(<4 x float>* %p, <4 x float>* %r) nounwind {
entry:
  %v = load <4 x float>* %p, align 1
  store <4 x float> %v, <4 x float>* %r, align 1
  ret void

; These loads and stores are legalized into aligned loads and stores
; using aligned stack slots.
; CHECK: @foo6
; CHECK-DAG: ld
; CHECK-DAG: ld
; CHECK-DAG: stdx
; CHECK: stdx
}

