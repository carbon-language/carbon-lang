; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.AMDGPU.umul24(i32, i32) nounwind readnone

; SI-LABEL: @test_umul24
define void @test_umul24(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %mul = call i32 @llvm.AMDGPU.umul24(i32 %src0, i32 %src1) nounwind readnone
  store i32 %mul, i32 addrspace(1)* %out, align 4
  ret void
}

