; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Optimize fneg to togglebit in V5.

define float @bar(float %x) nounwind {
entry:
; CHECK: r{{[0-9]+}} = togglebit(r{{[0-9]+}}, #31)
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %sub = fsub float -0.000000e+00, %0
  ret float %sub
}

define float @baz(float %x) nounwind {
entry:
; CHECK: r{{[0-9]+}} = togglebit(r{{[0-9]+}}, #31)
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %conv = fpext float %0 to double
  %mul = fmul double %conv, -1.000000e+00
  %conv1 = fptrunc double %mul to float
  ret float %conv1
}
