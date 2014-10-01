; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=r600 -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=rv770 -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.umad24(i32, i32, i32) nounwind readnone

; FUNC-LABEL: {{^}}test_umad24:
; SI: V_MAD_U32_U24
; EG: MULADD_UINT24
; R600: MULLO_UINT
; R600: ADD_INT
define void @test_umad24(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) nounwind {
  %mad = call i32 @llvm.AMDGPU.umad24(i32 %src0, i32 %src1, i32 %src2) nounwind readnone
  store i32 %mad, i32 addrspace(1)* %out, align 4
  ret void
}

