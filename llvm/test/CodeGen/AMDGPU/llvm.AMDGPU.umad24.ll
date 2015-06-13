; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=r600 -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=rv770 -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.umad24(i32, i32, i32) nounwind readnone
declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; FUNC-LABEL: {{^}}test_umad24:
; SI: v_mad_u32_u24
; EG: MULADD_UINT24
; R600: MULLO_UINT
; R600: ADD_INT
define void @test_umad24(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) nounwind {
  %mad = call i32 @llvm.AMDGPU.umad24(i32 %src0, i32 %src1, i32 %src2) nounwind readnone
  store i32 %mad, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}commute_umad24:
; SI-DAG: buffer_load_dword [[SRC0:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dword [[SRC2:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_mad_u32_u24 [[RESULT:v[0-9]+]], 4, [[SRC0]], [[SRC2]]
; SI: buffer_store_dword [[RESULT]]
define void @commute_umad24(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %src0.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %src2.gep = getelementptr i32, i32 addrspace(1)* %src0.gep, i32 1

  %src0 = load i32, i32 addrspace(1)* %src0.gep, align 4
  %src2 = load i32, i32 addrspace(1)* %src2.gep, align 4
  %mad = call i32 @llvm.AMDGPU.umad24(i32 %src0, i32 4, i32 %src2) nounwind readnone
  store i32 %mad, i32 addrspace(1)* %out.gep, align 4
  ret void
}

