;RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: @main
;CHECK: V_CMP_O_F32_e32 vcc, {{[sv][0-9]+, v[0-9]+}}

define void @main(float %p) {
main_body:
  %c = fcmp oeq float %p, %p
  %r = select i1 %c, float 1.000000e+00, float 0.000000e+00
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %r, float %r, float %r, float %r)
  ret void
}

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
