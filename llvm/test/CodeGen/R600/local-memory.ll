; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s


@local_memory.local_mem = internal addrspace(3) unnamed_addr global [16 x i32] zeroinitializer, align 4

; CHECK: @local_memory

; Check that the LDS size emitted correctly
; CHECK: .long 166120
; CHECK-NEXT: .long 16

; CHECK: LDS_WRITE

; GROUP_BARRIER must be the last instruction in a clause
; CHECK: GROUP_BARRIER
; CHECK-NEXT: ALU clause

; CHECK: LDS_READ_RET

define void @local_memory(i32 addrspace(1)* %out) {
entry:
  %y.i = call i32 @llvm.r600.read.tidig.x() #0
  %arrayidx = getelementptr inbounds [16 x i32] addrspace(3)* @local_memory.local_mem, i32 0, i32 %y.i
  store i32 %y.i, i32 addrspace(3)* %arrayidx, align 4
  %add = add nsw i32 %y.i, 1
  %cmp = icmp eq i32 %add, 16
  %.add = select i1 %cmp, i32 0, i32 %add
  call void @llvm.AMDGPU.barrier.local()
  %arrayidx1 = getelementptr inbounds [16 x i32] addrspace(3)* @local_memory.local_mem, i32 0, i32 %.add
  %0 = load i32 addrspace(3)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32 addrspace(1)* %out, i32 %y.i
  store i32 %0, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

@local_memory_two_objects.local_mem0 = internal addrspace(3) unnamed_addr global [4 x i32] zeroinitializer, align 4
@local_memory_two_objects.local_mem1 = internal addrspace(3) unnamed_addr global [4 x i32] zeroinitializer, align 4

; CHECK: @local_memory_two_objects

; Check that the LDS size emitted correctly
; CHECK: .long 166120
; CHECK-NEXT: .long 8

; Make sure the lds writes are using different addresses.
; CHECK: LDS_WRITE {{[*]*}} {{PV|T}}[[ADDRW:[0-9]*\.[XYZW]]]
; CHECK-NOT: LDS_WRITE {{[*]*}} T[[ADDRW]]

; GROUP_BARRIER must be the last instruction in a clause
; CHECK: GROUP_BARRIER
; CHECK-NEXT: ALU clause

; Make sure the lds reads are using different addresses.
; CHECK: LDS_READ_RET {{[*]*}} OQAP, {{PV|T}}[[ADDRR:[0-9]*\.[XYZW]]]
; CHECK-NOT: LDS_READ_RET {{[*]*}} OQAP, T[[ADDRR]]

define void @local_memory_two_objects(i32 addrspace(1)* %out) {
entry:
  %x.i = call i32 @llvm.r600.read.tidig.x() #0
  %arrayidx = getelementptr inbounds [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem0, i32 0, i32 %x.i
  store i32 %x.i, i32 addrspace(3)* %arrayidx, align 4
  %mul = shl nsw i32 %x.i, 1
  %arrayidx1 = getelementptr inbounds [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem1, i32 0, i32 %x.i
  store i32 %mul, i32 addrspace(3)* %arrayidx1, align 4
  %sub = sub nsw i32 3, %x.i
  call void @llvm.AMDGPU.barrier.local()
  %arrayidx2 = getelementptr inbounds [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem0, i32 0, i32 %sub
  %0 = load i32 addrspace(3)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32 addrspace(1)* %out, i32 %x.i
  store i32 %0, i32 addrspace(1)* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem1, i32 0, i32 %sub
  %1 = load i32 addrspace(3)* %arrayidx4, align 4
  %add = add nsw i32 %x.i, 4
  %arrayidx5 = getelementptr inbounds i32 addrspace(1)* %out, i32 %add
  store i32 %1, i32 addrspace(1)* %arrayidx5, align 4
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #0
declare void @llvm.AMDGPU.barrier.local()

attributes #0 = { readnone }
