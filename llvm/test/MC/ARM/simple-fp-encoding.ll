;RUN: llc -mtriple=armv7-apple-darwin -mcpu=cortex-a8 -mattr=-neonfp -show-mc-encoding < %s | FileCheck %s


; FIXME: Once the ARM integrated assembler is up and going, these sorts of tests
;        should run on .s source files rather than using llc to generate the
;        assembly.


define arm_aapcscc float @f1(float %a, float %b) nounwind {
entry:
; CHECK: f1
; CHECK: vadd.f32 s0, s1, s0  @ encoding: [0x80,0x0a,0x30,0xee]
  %add = fadd float %a, %b
  ret float %add
}

define arm_aapcscc double @f2(double %a, double %b) nounwind {
entry:
; CHECK: f2
; CHECK: vadd.f64 d16, d17, d16  @ encoding: [0xa0,0x0b,0x71,0xee]
  %add = fadd double %a, %b
  ret double %add
}
