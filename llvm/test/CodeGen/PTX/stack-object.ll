; RUN: llc < %s -march=ptx32 -mattr=sm20 | FileCheck %s

define ptx_device float @stack1(float %a) {
  ; CHECK: .local .align 4 .b8 __local0[4];
  %a.2 = alloca float, align 4
  ; CHECK: st.local.f32 [__local0], %f0
  store float %a, float* %a.2
  %a.3 = load float* %a.2
  ret float %a.3
}

define ptx_device float @stack1_align8(float %a) {
  ; CHECK: .local .align 8 .b8 __local0[4];
  %a.2 = alloca float, align 8
  ; CHECK: st.local.f32 [__local0], %f0
  store float %a, float* %a.2
  %a.3 = load float* %a.2
  ret float %a.3
}
