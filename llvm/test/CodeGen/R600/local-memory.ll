; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=bonaire -verify-machineinstrs | FileCheck --check-prefix=CI-CHECK %s

@local_memory.local_mem = internal addrspace(3) unnamed_addr global [128 x i32] zeroinitializer, align 4

; EG-CHECK-LABEL: @local_memory
; SI-CHECK-LABEL: @local_memory
; CI-CHECK-LABEL: @local_memory

; Check that the LDS size emitted correctly
; EG-CHECK: .long 166120
; EG-CHECK-NEXT: .long 128
; SI-CHECK: .long 47180
; SI-CHECK-NEXT: .long 65536
; CI-CHECK: .long 47180
; CI-CHECK-NEXT: .long 32768

; EG-CHECK: LDS_WRITE
; SI-CHECK-NOT: S_WQM_B64
; SI-CHECK: DS_WRITE_B32 0

; GROUP_BARRIER must be the last instruction in a clause
; EG-CHECK: GROUP_BARRIER
; EG-CHECK-NEXT: ALU clause
; SI-CHECK: S_BARRIER

; EG-CHECK: LDS_READ_RET
; SI-CHECK: DS_READ_B32 {{v[0-9]+}}, 0

define void @local_memory(i32 addrspace(1)* %out) {
entry:
  %y.i = call i32 @llvm.r600.read.tidig.x() #0
  %arrayidx = getelementptr inbounds [128 x i32] addrspace(3)* @local_memory.local_mem, i32 0, i32 %y.i
  store i32 %y.i, i32 addrspace(3)* %arrayidx, align 4
  %add = add nsw i32 %y.i, 1
  %cmp = icmp eq i32 %add, 16
  %.add = select i1 %cmp, i32 0, i32 %add
  call void @llvm.AMDGPU.barrier.local()
  %arrayidx1 = getelementptr inbounds [128 x i32] addrspace(3)* @local_memory.local_mem, i32 0, i32 %.add
  %0 = load i32 addrspace(3)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32 addrspace(1)* %out, i32 %y.i
  store i32 %0, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #0
declare void @llvm.AMDGPU.barrier.local()

attributes #0 = { readnone }
