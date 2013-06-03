;RUN: llc < %s -march=r600 -mcpu=verde | FileCheck %s

;CHECK: V_CMP_U_F32_e64 SGPR0_SGPR1, {{[SV]GPR[0-9]+, [SV]GPR[0-9]+}}, 0, 0, 0, 0

define void @main(float %p) {
main_body:
  %c = fcmp une float %p, %p
  %r = select i1 %c, float 1.000000e+00, float 0.000000e+00
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %r, float %r, float %r, float %r)
  ret void
}

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
