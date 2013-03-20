; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; Even though general vector types are not supported in PTX, we can still
; optimize loads/stores with pseudo-vector instructions of the form:
;
; ld.v2.f32 {%f0, %f1}, [%r0]
;
; which will load two floats at once into scalar registers.

define void @foo(<2 x float>* %a) {
; CHECK: .func foo
; CHECK: ld.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}
  %t1 = load <2 x float>* %a
  %t2 = fmul <2 x float> %t1, %t1
  store <2 x float> %t2, <2 x float>* %a
  ret void
}

define void @foo2(<4 x float>* %a) {
; CHECK: .func foo2
; CHECK: ld.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  %t1 = load <4 x float>* %a
  %t2 = fmul <4 x float> %t1, %t1
  store <4 x float> %t2, <4 x float>* %a
  ret void
}

define void @foo3(<8 x float>* %a) {
; CHECK: .func foo3
; CHECK: ld.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
; CHECK-NEXT: ld.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  %t1 = load <8 x float>* %a
  %t2 = fmul <8 x float> %t1, %t1
  store <8 x float> %t2, <8 x float>* %a
  ret void
}



define void @foo4(<2 x i32>* %a) {
; CHECK: .func foo4
; CHECK: ld.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = load <2 x i32>* %a
  %t2 = mul <2 x i32> %t1, %t1
  store <2 x i32> %t2, <2 x i32>* %a
  ret void
}

define void @foo5(<4 x i32>* %a) {
; CHECK: .func foo5
; CHECK: ld.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = load <4 x i32>* %a
  %t2 = mul <4 x i32> %t1, %t1
  store <4 x i32> %t2, <4 x i32>* %a
  ret void
}

define void @foo6(<8 x i32>* %a) {
; CHECK: .func foo6
; CHECK: ld.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
; CHECK-NEXT: ld.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = load <8 x i32>* %a
  %t2 = mul <8 x i32> %t1, %t1
  store <8 x i32> %t2, <8 x i32>* %a
  ret void
}
