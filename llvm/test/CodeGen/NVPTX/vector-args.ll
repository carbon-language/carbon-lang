; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s


define float @foo(<2 x float> %a) {
; CHECK: .func (.param .b32 func_retval0) foo
; CHECK: .param .align 8 .b8 foo_param_0[8]
; CHECK: ld.param.f32 %f{{[0-9]+}}
; CHECK: ld.param.f32 %f{{[0-9]+}}
  %t1 = fmul <2 x float> %a, %a
  %t2 = extractelement <2 x float> %t1, i32 0
  %t3 = extractelement <2 x float> %t1, i32 1
  %t4 = fadd float %t2, %t3
  ret float %t4
}


define float @bar(<4 x float> %a) {
; CHECK: .func (.param .b32 func_retval0) bar
; CHECK: .param .align 16 .b8 bar_param_0[16]
; CHECK: ld.param.f32 %f{{[0-9]+}}
; CHECK: ld.param.f32 %f{{[0-9]+}}
  %t1 = fmul <4 x float> %a, %a
  %t2 = extractelement <4 x float> %t1, i32 0
  %t3 = extractelement <4 x float> %t1, i32 1
  %t4 = fadd float %t2, %t3
  ret float %t4
}
