; RUN: llc < %s -march=nvptx -mcpu=sm_20 -fp-contract=fast | FileCheck %s

define ptx_device float @t1_f32(float %x, float %y, float %z,
                                float %u, float %v) {
; CHECK: fma.rn.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}};
; CHECK: fma.rn.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}};
; CHECK: ret;
  %a = fmul float %x, %y
  %b = fmul float %u, %v
  %c = fadd float %a, %b
  %d = fadd float %c, %z
  ret float %d
}

define ptx_device double @t1_f64(double %x, double %y, double %z,
                                 double %u, double %v) {
; CHECK: fma.rn.f64 %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}};
; CHECK: fma.rn.f64 %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}};
; CHECK: ret;
  %a = fmul double %x, %y
  %b = fmul double %u, %v
  %c = fadd double %a, %b
  %d = fadd double %c, %z
  ret double %d
}
