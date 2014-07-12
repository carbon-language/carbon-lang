; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: @test_barrier_global
; EG: GROUP_BARRIER
; SI: S_BARRIER

define void @test_barrier_global(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x()
  %1 = getelementptr i32 addrspace(1)* %out, i32 %0
  store i32 %0, i32 addrspace(1)* %1
  call void @llvm.AMDGPU.barrier.global()
  %2 = call i32 @llvm.r600.read.local.size.x()
  %3 = sub i32 %2, 1
  %4 = sub i32 %3, %0
  %5 = getelementptr i32 addrspace(1)* %out, i32 %4
  %6 = load i32 addrspace(1)* %5
  store i32 %6, i32 addrspace(1)* %1
  ret void
}

declare void @llvm.AMDGPU.barrier.global()

declare i32 @llvm.r600.read.tidig.x() #0
declare i32 @llvm.r600.read.local.size.x() #0

attributes #0 = { readnone }
