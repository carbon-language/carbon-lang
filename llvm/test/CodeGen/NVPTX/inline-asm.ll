; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s


define float @test(float %x) {
entry:
; CHECK: ex2.approx.ftz.f32 %f{{[0-9]+}}, %f{{[0-9]+}}
  %0 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=f,f"(float %x)
  ret float %0
}
