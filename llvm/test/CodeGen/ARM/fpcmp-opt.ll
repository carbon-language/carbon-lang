; RUN: llc < %s -march=arm -mcpu=cortex-a8 -mattr=+vfp2 -enable-unsafe-fp-math -enable-no-nans-fp-math | FileCheck -check-prefix=FINITE %s
; RUN: llc < %s -march=arm -mcpu=cortex-a8 -mattr=+vfp2 -enable-unsafe-fp-math | FileCheck -check-prefix=NAN %s
; rdar://7461510

define arm_apcscc i32 @t1(float* %a, float* %b) nounwind {
entry:
; FINITE: t1:
; FINITE-NOT: vldr
; FINITE: ldr
; FINITE: ldr
; FINITE: cmp r0, r1
; FINITE-NOT: vcmpe.f32
; FINITE-NOT: vmrs
; FINITE: beq

; NAN: t1:
; NAN: vldr.32 s0,
; NAN: vldr.32 s1,
; NAN: vcmpe.f32 s1, s0
; NAN: vmrs apsr_nzcv, fpscr
; NAN: beq
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

define arm_apcscc i32 @t2(double* %a, double* %b) nounwind {
entry:
; FINITE: t2:
; FINITE-NOT: vldr
; FINITE: ldrd r0, [r0]
; FINITE: cmp r0, #0
; FINITE: cmpeq r1, #0
; FINITE-NOT: vcmpe.f32
; FINITE-NOT: vmrs
; FINITE: bne
  %0 = load double* %a
  %1 = fcmp oeq double %0, 0.000000e+00
  br i1 %1, label %bb1, label %bb2

bb1:
  %2 = call i32 @bar()
  ret i32 %2

bb2:
  %3 = call i32 @foo()
  ret i32 %3
}

define arm_apcscc i32 @t3(float* %a, float* %b) nounwind {
entry:
; FINITE: t3:
; FINITE-NOT: vldr
; FINITE: ldr r0, [r0]
; FINITE: cmp r0, #0
; FINITE-NOT: vcmpe.f32
; FINITE-NOT: vmrs
; FINITE: bne
  %0 = load float* %a
  %1 = fcmp oeq float %0, 0.000000e+00
  br i1 %1, label %bb1, label %bb2

bb1:
  %2 = call i32 @bar()
  ret i32 %2

bb2:
  %3 = call i32 @foo()
  ret i32 %3
}

declare i32 @bar()
declare i32 @foo()
