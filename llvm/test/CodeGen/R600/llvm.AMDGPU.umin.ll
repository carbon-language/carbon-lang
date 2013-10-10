;RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck %s

;CHECK: V_MIN_U32_e32

define void @main(i32 %p0, i32 %p1) #0 {
main_body:
  %0 = call i32 @llvm.AMDGPU.umin(i32 %p0, i32 %p1)
  %1 = bitcast i32 %0 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %1, float %1, float %1, float %1)
  ret void
}

; Function Attrs: readnone
declare i32 @llvm.AMDGPU.umin(i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { readnone }

!0 = metadata !{metadata !"const", null, i32 1}
