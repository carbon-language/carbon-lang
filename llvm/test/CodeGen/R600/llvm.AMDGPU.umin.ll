; RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck -check-prefix=SI %s

; SI-LABEL: @vector_umin
; SI: V_MIN_U32_e32
define void @vector_umin(i32 %p0, i32 %p1, i32 addrspace(1)* %in) #0 {
main_body:
  %load = load i32 addrspace(1)* %in, align 4
  %min = call i32 @llvm.AMDGPU.umin(i32 %p0, i32 %load)
  %bc = bitcast i32 %min to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %bc, float %bc, float %bc, float %bc)
  ret void
}

; SI-LABEL: @scalar_umin
; SI: S_MIN_U32
define void @scalar_umin(i32 %p0, i32 %p1) #0 {
entry:
  %min = call i32 @llvm.AMDGPU.umin(i32 %p0, i32 %p1)
  %bc = bitcast i32 %min to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %bc, float %bc, float %bc, float %bc)
  ret void
}

; Function Attrs: readnone
declare i32 @llvm.AMDGPU.umin(i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!0 = metadata !{metadata !"const", null, i32 1}
