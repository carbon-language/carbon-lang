; RUN: llc < %s -march=arm -mattr=+vfp2 -enable-unsafe-fp-math | FileCheck %s
; rdar://7461510

define arm_apcscc i32 @t1(float* %a, float* %b) nounwind {
entry:
; CHECK: t1:
; CHECK-NOT: vldr
; CHECK: ldr
; CHECK: ldr
; CHECK: cmp r0, r1
; CHECK-NOT: vcmpe.f32
; CHECK-NOT: vmrs
; CHECK: beq
  %0 = load float* %a
  %1 = load float* %b
  %2 = fcmp une float %0, %1
  br i1 %2, label %bb1, label %bb2

bb1:
  %3 = call i32 @bar()
  ret i32 %3

bb2:
  %4 = call i32 @foo()
  ret i32 %4
}

declare i32 @bar()
declare i32 @foo()
