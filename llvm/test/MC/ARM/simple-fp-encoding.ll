; RUN: llc -mtriple=armv7-apple-darwin -mcpu=cortex-a8 -mattr=-neonfp -show-mc-encoding < %s | FileCheck %s

; XFAIL: *

; FIXME: Once the ARM integrated assembler is up and going, these sorts of tests
;        should run on .s source files rather than using llc to generate the
;        assembly.


define double @f1(double %a, double %b) nounwind readnone {
entry:
; CHECK: f1
; CHECK: vadd.f64 d16, d17, d16      @ encoding: [0xa0,0x0b,0x71,0xee]
  %add = fadd double %a, %b
  ret double %add
}

define float @f2(float %a, float %b) nounwind readnone {
entry:
; CHECK: f2
; CHECK: vadd.f32 s0, s1, s0         @ encoding: [0x80,0x0a,0x30,0xee]
  %add = fadd float %a, %b
  ret float %add
}

define double @f3(double %a, double %b) nounwind readnone {
entry:
; CHECK: f3
; CHECK: vsub.f64 d16, d17, d16      @ encoding: [0xe0,0x0b,0x71,0xee]
  %sub = fsub double %a, %b
  ret double %sub
}

define float @f4(float %a, float %b) nounwind readnone {
entry:
; CHECK: f4
; CHECK: vsub.f32 s0, s1, s0         @ encoding: [0xc0,0x0a,0x30,0xee]
  %sub = fsub float %a, %b
  ret float %sub
}

define double @f5(double %a, double %b) nounwind readnone {
entry:
; CHECK: f5
; CHECK: vdiv.f64 d16, d17, d16      @ encoding: [0xa0,0x0b,0xc1,0xee]
  %div = fdiv double %a, %b
  ret double %div
}

define float @f6(float %a, float %b) nounwind readnone {
entry:
; CHECK: f6
; CHECK: vdiv.f32 s0, s1, s0         @ encoding: [0x80,0x0a,0x80,0xee]
  %div = fdiv float %a, %b
  ret float %div
}

define double @f7(double %a, double %b) nounwind readnone {
entry:
; CHECK: f7
; CHECK: vmul.f64 d16, d17, d16      @ encoding: [0xa0,0x0b,0x61,0xee]
  %mul = fmul double %a, %b
  ret double %mul
}

define float @f8(float %a, float %b) nounwind readnone {
entry:
; CHECK: f8
; CHECK: vmul.f32 s0, s1, s0         @ encoding: [0x80,0x0a,0x20,0xee]
  %mul = fmul float %a, %b
  ret float %mul
}

define double @f9(double %a, double %b) nounwind readnone {
entry:
; CHECK: f9
; CHECK: vnmul.f64 d16, d17, d16     @ encoding: [0xe0,0x0b,0x61,0xee]
  %mul = fmul double %a, %b
  %sub = fsub double -0.000000e+00, %mul
  ret double %sub
}

define void @f10(float %a, float %b, float* %c) nounwind readnone {
entry:
; CHECK: f10
; CHECK: vnmul.f32 s0, s1, s0        @ encoding: [0xc0,0x0a,0x20,0xee]
  %mul = fmul float %a, %b
  %sub = fsub float -0.000000e+00, %mul
  store float %sub, float* %c, align 4
  ret void
}

define i1 @f11(double %a, double %b) nounwind readnone {
entry:
; CHECK: f11
; CHECK: vcmpe.f64 d17, d16          @ encoding: [0xe0,0x1b,0xf4,0xee]
  %cmp = fcmp oeq double %a, %b
  ret i1 %cmp
}

define i1 @f12(float %a, float %b) nounwind readnone {
entry:
; CHECK: f12
; CHECK: vcmpe.f32 s1, s0            @ encoding: [0xc0,0x0a,0xf4,0xee]
  %cmp = fcmp oeq float %a, %b
  ret i1 %cmp
}

define i1 @f13(double %a) nounwind readnone {
entry:
; CHECK: f13
; CHECK: vcmpe.f64 d16, #0           @ encoding: [0xc0,0x0b,0xf5,0xee]
  %cmp = fcmp oeq double %a, 0.000000e+00
  ret i1 %cmp
}

define i1 @f14(float %a) nounwind readnone {
entry:
; CHECK: f14
; CHECK: vcmpe.f32 s0, #0            @ encoding: [0xc0,0x0a,0xb5,0xee]
  %cmp = fcmp oeq float %a, 0.000000e+00
  ret i1 %cmp
}

define double @f15(double %a) nounwind {
entry:
; CHECK: f15
; CHECK: vabs.f64 d16, d16           @ encoding: [0xe0,0x0b,0xf0,0xee]
  %call = tail call double @fabsl(double %a)
  ret double %call
}

declare double @fabsl(double)

define float @f16(float %a) nounwind {
entry:
; CHECK: f16
; FIXME: This call generates a "bfc" instruction instead of "vabs.f32".
  %call = tail call float @fabsf(float %a)
  ret float %call
}

declare float @fabsf(float)

define float @f17(double %a) nounwind readnone {
entry:
; CHECK: f17
; CHECK: vcvt.f32.f64 s0, d16        @ encoding: [0xe0,0x0b,0xb7,0xee]
  %conv = fptrunc double %a to float
  ret float %conv
}

define double @f18(float %a) nounwind readnone {
entry:
; CHECK: f18
; CHECK: vcvt.f64.f32 d16, s0        @ encoding: [0xc0,0x0a,0xf7,0xee]
  %conv = fpext float %a to double
  ret double %conv
}

define double @f19(double %a) nounwind readnone {
entry:
; CHECK: f19
; CHECK: vneg.f64 d16, d16           @ encoding: [0x60,0x0b,0xf1,0xee]
  %sub = fsub double -0.000000e+00, %a
  ret double %sub
}

define float @f20(float %a) nounwind readnone {
entry:
; CHECK: f20
; FIXME: This produces an 'eor' instruction.
  %sub = fsub float -0.000000e+00, %a
  ret float %sub
}

define double @f21(double %a) nounwind readnone {
entry:
; CHECK: f21
; CHECK: vsqrt.f64 d16, d16          @ encoding: [0xe0,0x0b,0xf1,0xee]
  %call = tail call double @sqrtl(double %a) nounwind
  ret double %call
}

declare double @sqrtl(double) readnone

define float @f22(float %a) nounwind readnone {
entry:
; CHECK: f22
; CHECK: vsqrt.f32 s0, s0            @ encoding: [0xc0,0x0a,0xb1,0xee]
  %call = tail call float @sqrtf(float %a) nounwind
  ret float %call
}

declare float @sqrtf(float) readnone

define double @f23(i32 %a) nounwind readnone {
entry:
; CHECK: f23
; CHECK: vcvt.f64.s32 d16, s0        @ encoding: [0xc0,0x0b,0xf8,0xee]
  %conv = sitofp i32 %a to double
  ret double %conv
}

define float @f24(i32 %a) nounwind readnone {
entry:
; CHECK: f24
; CHECK: vcvt.f32.s32 s0, s0         @ encoding: [0xc0,0x0a,0xb8,0xee]
  %conv = sitofp i32 %a to float
  ret float %conv
}

define double @f25(i32 %a) nounwind readnone {
entry:
; CHECK: f25
; CHECK: vcvt.f64.u32 d16, s0        @ encoding: [0x40,0x0b,0xf8,0xee]
  %conv = uitofp i32 %a to double
  ret double %conv
}

define float @f26(i32 %a) nounwind readnone {
entry:
; CHECK: f26
; CHECK: vcvt.f32.u32 s0, s0         @ encoding: [0x40,0x0a,0xb8,0xee]
  %conv = uitofp i32 %a to float
  ret float %conv
}

define i32 @f27(double %a) nounwind readnone {
entry:
; CHECK: f27
; CHECK: vcvt.s32.f64 s0, d16        @ encoding: [0xe0,0x0b,0xbd,0xee]
  %conv = fptosi double %a to i32
  ret i32 %conv
}

define i32 @f28(float %a) nounwind readnone {
entry:
; CHECK: f28
; CHECK: vcvt.s32.f32 s0, s0         @ encoding: [0xc0,0x0a,0xbd,0xee]
  %conv = fptosi float %a to i32
  ret i32 %conv
}

define i32 @f29(double %a) nounwind readnone {
entry:
; CHECK: f29
; CHECK: vcvt.u32.f64 s0, d16        @ encoding: [0xe0,0x0b,0xbc,0xee]
  %conv = fptoui double %a to i32
  ret i32 %conv
}

define i32 @f30(float %a) nounwind readnone {
entry:
; CHECK: f30
; CHECK: vcvt.u32.f32 s0, s0         @ encoding: [0xc0,0x0a,0xbc,0xee]
  %conv = fptoui float %a to i32
  ret i32 %conv
}

define double @f90(double %a, double %b, double %c) nounwind readnone {
entry:
; CHECK: f90
; FIXME: vmla.f64 d16, d18, d17      @ encoding: [0xa1,0x0b,0x42,0xee]
  %mul = fmul double %a, %b
  %add = fadd double %mul, %c
  ret double %add
}

define float @f91(float %a, float %b, float %c) nounwind readnone {
entry:
; CHECK: f91
; CHECK: vmla.f32 s1, s2, s0         @ encoding: [0x00,0x0a,0x41,0xee]
  %mul = fmul float %a, %b
  %add = fadd float %mul, %c
  ret float %add
}

define double @f92(double %a, double %b, double %c) nounwind readnone {
entry:
; CHECK: f92
; CHECK: vmls.f64 d16, d18, d17      @ encoding: [0xe1,0x0b,0x42,0xee]
  %mul = fmul double %a, %b
  %sub = fsub double %c, %mul
  ret double %sub
}

define float @f93(float %a, float %b, float %c) nounwind readnone {
entry:
; CHECK: f93
; CHECK: vmls.f32 s1, s2, s0         @ encoding: [0x40,0x0a,0x41,0xee]
  %mul = fmul float %a, %b
  %sub = fsub float %c, %mul
  ret float %sub
}

define double @f94(double %a, double %b, double %c) nounwind readnone {
entry:
; CHECK: f94
; CHECK: vnmla.f64 d16, d18, d17     @ encoding: [0xe1,0x0b,0x52,0xee]
  %mul = fmul double %a, %b
  %sub = fsub double -0.000000e+00, %mul
  %sub3 = fsub double %sub, %c
  ret double %sub3
}

define float @f95(float %a, float %b, float %c) nounwind readnone {
entry:
; CHECK: f95
; CHECK: vnmla.f32 s1, s2, s0        @ encoding: [0x40,0x0a,0x51,0xee]
  %mul = fmul float %a, %b
  %sub = fsub float -0.000000e+00, %mul
  %sub3 = fsub float %sub, %c
  ret float %sub3
}

define double @f96(double %a, double %b, double %c) nounwind readnone {
entry:
; CHECK: f96
; CHECK: vnmls.f64 d16, d18, d17     @ encoding: [0xa1,0x0b,0x52,0xee]
  %mul = fmul double %a, %b
  %sub = fsub double %mul, %c
  ret double %sub
}

define float @f97(float %a, float %b, float %c) nounwind readnone {
entry:
; CHECK: f97
; CHECK: vnmls.f32 s1, s2, s0        @ encoding: [0x00,0x0a,0x51,0xee]
  %mul = fmul float %a, %b
  %sub = fsub float %mul, %c
  ret float %sub
}

; FIXME: Check for fmstat instruction.


define double @f98(double %a, i32 %i) nounwind readnone {
entry:
  %cmp = icmp eq i32 %i, 3
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
; CHECK: f98
; CHECK: vnegne.f64 d16, d16         @ encoding: [0x60,0x0b,0xf1,0x1e]
  %sub = fsub double -0.000000e+00, %a
  ret double %sub

return:                                           ; preds = %entry
  ret double %a
}

define float @f99(float %a, float %b, i32 %i) nounwind readnone {
entry:
  %cmp = icmp eq i32 %i, 3
  br i1 %cmp, label %if.end, label %return

if.end:                                           ; preds = %entry
; CHECK: f99
; CHECK: vmovne s0, r0               @ encoding: [0x10,0x0a,0x00,0x1e]
; CHECK: vmoveq s0, r1               @ encoding: [0x10,0x1a,0x00,0x0e]
  ret float %b

return:                                           ; preds = %entry
  ret float %a
}


define i32 @f100() nounwind readnone {
entry:
; CHECK: f100
; CHECK: vmrs r0, fpscr              @ encoding: [0x10,0x0a,0xf1,0xee]
  %0 = tail call i32 @llvm.arm.get.fpscr()
  ret i32 %0
}

declare i32 @llvm.arm.get.fpscr() nounwind readnone

define void @f101(i32 %a) nounwind {
entry:
; CHECK: f101
; CHECK: vmsr fpscr, r0              @ encoding: [0x10,0x0a,0xe1,0xee]
  tail call void @llvm.arm.set.fpscr(i32 %a)
  ret void
}

declare void @llvm.arm.set.fpscr(i32) nounwind


define double @f102() nounwind readnone {
entry:
; CHECK: f102
; CHECK: vmov.f64 d16, #3.000000e+00 @ encoding: [0x08,0x0b,0xf0,0xee]
  ret double 3.000000e+00
}

define float @f103(float %a) nounwind readnone {
entry:
; CHECK: f103
; CHECK: vmov.f32 s0, #3.000000e+00  @ encoding: [0x08,0x0a,0xb0,0xee]
  %add = fadd float %a, 3.000000e+00
  ret float %add
}

define void @f104(float %a, float %b, float %c, float %d, float %e, float %f) nounwind {
entry:
; CHECK: f104
; CHECK: vmov s0, r0                 @ encoding: [0x10,0x0a,0x00,0xee]
; CHECK: vmov s1, r1                 @ encoding: [0x90,0x1a,0x00,0xee]
; CHECK: vmov s2, r2                 @ encoding: [0x10,0x2a,0x01,0xee]
; CHECK: vmov s3, r3                 @ encoding: [0x90,0x3a,0x01,0xee]
  %conv = fptosi float %a to i32
  %conv2 = fptosi float %b to i32
  %conv4 = fptosi float %c to i32
  %conv6 = fptosi float %d to i32
  %conv8 = fptosi float %e to i32
  %conv10 = fptosi float %f to i32
  tail call void @g104(i32 %conv, i32 %conv2, i32 %conv4, i32 %conv6, i32 %conv8, i32 %conv10) nounwind
; CHECK: vmov r0, s0                 @ encoding: [0x10,0x0a,0x10,0xee]
; CHECK: vmov r1, s1                 @ encoding: [0x90,0x1a,0x10,0xee]
; CHECK: vmov r2, s2                 @ encoding: [0x10,0x2a,0x11,0xee]
; CHECK: vmov r3, s3                 @ encoding: [0x90,0x3a,0x11,0xee]
  ret void
}

declare void @g104(i32, i32, i32, i32, i32, i32)

define double @f105(i32 %a) nounwind readnone {
entry:
; CHECK: f105
; CHECK: vmov r0, r1, d16            @ encoding: [0x30,0x0b,0x51,0xec]
  %conv = uitofp i32 %a to double
  ret double %conv
}
