;RUN: llc < %s -march=r600 -mcpu=verde | FileCheck %s

;CHECK: V_CVT_F32_U32_e32

define void @main(i32 %p) #0 {
main_body:
  %0 = uitofp i32 %p to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %0, float %0, float %0, float %0)
  ret void
}

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }

!0 = metadata !{metadata !"const", null, i32 1}
