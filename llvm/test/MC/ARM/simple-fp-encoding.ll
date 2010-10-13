;RUN: llc -mtriple=armv7-apple-darwin -mcpu=cortex-a8 -mattr=-neonfp -show-mc-encoding < %s | FileCheck %s


; FIXME: Once the ARM integrated assembler is up and going, these sorts of tests
;        should run on .s source files rather than using llc to generate the
;        assembly.


define double @f1(double %a, double %b) nounwind readnone {
entry:
; CHECK: f1
; CHECK: vadd.f64 d16, d17, d16  @ encoding: [0xa0,0x0b,0x71,0xee]
  %add = fadd double %a, %b
  ret double %add
}

define float @f2(float %a, float %b) nounwind readnone {
entry:
; CHECK: f2
; CHECK: vadd.f32 s0, s1, s0  @ encoding: [0x80,0x0a,0x30,0xee]
  %add = fadd float %a, %b
  ret float %add
}

define double @f3(double %a, double %b) nounwind readnone {
entry:
; CHECK: f3
; CHECK: vsub.f64 d16, d17, d16  @ encoding: [0xe0,0x0b,0x71,0xee]
  %sub = fsub double %a, %b
  ret double %sub
}

define float @f4(float %a, float %b) nounwind readnone {
entry:
; CHECK: f4
; CHECK: vsub.f32 s0, s1, s0  @ encoding: [0xc0,0x0a,0x30,0xee]
  %sub = fsub float %a, %b
  ret float %sub
}

define double @f5(double %a, double %b) nounwind readnone {
entry:
; CHECK: f5
; CHECK: vdiv.f64 d16, d17, d16  @ encoding: [0xa0,0x0b,0xc1,0xee]
  %div = fdiv double %a, %b
  ret double %div
}

define float @f6(float %a, float %b) nounwind readnone {
entry:
; CHECK: f6
; CHECK: vdiv.f32 s0, s1, s0  @ encoding: [0x80,0x0a,0x80,0xee]
  %div = fdiv float %a, %b
  ret float %div
}

define double @f7(double %a, double %b) nounwind readnone {
entry:
; CHECK: f7
; CHECK: vmul.f64 d16, d17, d16  @ encoding: [0xa0,0x0b,0x61,0xee]
  %mul = fmul double %a, %b
  ret double %mul
}

define float @f8(float %a, float %b) nounwind readnone {
entry:
; CHECK: f8
; CHECK: vmul.f32 s0, s1, s0  @ encoding: [0x80,0x0a,0x20,0xee]
  %mul = fmul float %a, %b
  ret float %mul
}

define double @f9(double %a, double %b) nounwind readnone {
entry:
; CHECK: f9
; CHECK: vnmul.f64 d16, d17, d16  @ encoding: [0xe0,0x0b,0x61,0xee]
  %mul = fmul double %a, %b
  %sub = fsub double -0.000000e+00, %mul
  ret double %sub
}

define void @f10(float %a, float %b, float* %c) nounwind readnone {
entry:
; CHECK: f10
; CHECK: vnmul.f32 s0, s1, s0  @ encoding: [0xc0,0x0a,0x20,0xee]
  %mul = fmul float %a, %b
  %sub = fsub float -0.000000e+00, %mul
  store float %sub, float* %c, align 4
  ret void
}

define i1 @f11(double %a, double %b) nounwind readnone {
entry:
; CHECK: f11
; CHECK: vcmpe.f64 d17, d16  @ encoding: [0xe0,0x1b,0xf4,0xee]
  %cmp = fcmp oeq double %a, %b
  ret i1 %cmp
}

define i1 @f12(float %a, float %b) nounwind readnone {
entry:
; CHECK: f12
; CHECK: vcmpe.f32 s1, s0  @ encoding: [0xc0,0x0a,0xf4,0xee]
  %cmp = fcmp oeq float %a, %b
  ret i1 %cmp
}
