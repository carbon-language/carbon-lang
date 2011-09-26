; RUN: llc < %s -march=ptx32 -mattr=sm20 | FileCheck %s

define ptx_device float @stack1(float %a) {
  ; CHECK: .local .b32 __local0;
  %a.2 = alloca float
  ; CHECK: st.local.f32 [__local0], %f0
  store float %a, float* %a.2
  %a.3 = load float* %a.2
  ret float %a.3
}
