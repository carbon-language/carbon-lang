; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI-CHECK %s

@local_memory_two_objects.local_mem0 = internal addrspace(3) unnamed_addr global [4 x i32] zeroinitializer, align 4
@local_memory_two_objects.local_mem1 = internal addrspace(3) unnamed_addr global [4 x i32] zeroinitializer, align 4

; EG-CHECK: @local_memory_two_objects

; Check that the LDS size emitted correctly
; EG-CHECK: .long 166120
; EG-CHECK-NEXT: .long 8
; SI-CHECK: .long 47180
; SI-CHECK-NEXT: .long 32768

; Make sure the lds writes are using different addresses.
; EG-CHECK: LDS_WRITE {{[*]*}} {{PV|T}}[[ADDRW:[0-9]*\.[XYZW]]]
; EG-CHECK-NOT: LDS_WRITE {{[*]*}} T[[ADDRW]]
; SI-CHECK: DS_WRITE_B32 0, {{VGPR[0-9]*}}, VGPR[[ADDRW:[0-9]*]]
; SI-CHECK-NOT: DS_WRITE_B32 0, {{VGPR[0-9]*}}, VGPR[[ADDRW]]

; GROUP_BARRIER must be the last instruction in a clause
; EG-CHECK: GROUP_BARRIER
; EG-CHECK-NEXT: ALU clause

; Make sure the lds reads are using different addresses.
; EG-CHECK: LDS_READ_RET {{[*]*}} OQAP, {{PV|T}}[[ADDRR:[0-9]*\.[XYZW]]]
; EG-CHECK-NOT: LDS_READ_RET {{[*]*}} OQAP, T[[ADDRR]]
; SI-CHECK: DS_READ_B32 {{VGPR[0-9]+}}, 0, [[ADDRR:VGPR[0-9]+]]
; SI-CHECK-NOT: DS_READ_B32 {{VGPR[0-9]+}}, 0, [[ADDRR]]

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
