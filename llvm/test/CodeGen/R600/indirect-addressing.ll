; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; This test checks that uses and defs of the AR register happen in the same
; instruction clause.

; CHECK: @mova_same_clause
; CHECK: MOVA_INT
; CHECK-NOT: ALU clause
; CHECK: 0 + AR.x
; CHECK: MOVA_INT
; CHECK-NOT: ALU clause
; CHECK: 0 + AR.x

define void @mova_same_clause(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) {
entry:
  %stack = alloca [5 x i32], align 4
  %0 = load i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32]* %stack, i32 0, i32 %0
  store i32 4, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32 addrspace(1)* %in, i32 1
  %1 = load i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32]* %stack, i32 0, i32 %1
  store i32 5, i32* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32]* %stack, i32 0, i32 0
  %2 = load i32* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32]* %stack, i32 0, i32 1
  %3 = load i32* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

; This test checks that the stack offset is calculated correctly for structs.
; All register loads/stores should be optimized away, so there shouldn't be
; any MOVA instructions.
;
; XXX: This generated code has unnecessary MOVs, we should be able to optimize
; this.

; CHECK: @multiple_structs
; CHECK-NOT: MOVA_INT

%struct.point = type { i32, i32 }

define void @multiple_structs(i32 addrspace(1)* %out) {
entry:
  %a = alloca %struct.point
  %b = alloca %struct.point
  %a.x.ptr = getelementptr %struct.point* %a, i32 0, i32 0
  %a.y.ptr = getelementptr %struct.point* %a, i32 0, i32 1
  %b.x.ptr = getelementptr %struct.point* %b, i32 0, i32 0
  %b.y.ptr = getelementptr %struct.point* %b, i32 0, i32 1
  store i32 0, i32* %a.x.ptr
  store i32 1, i32* %a.y.ptr
  store i32 2, i32* %b.x.ptr
  store i32 3, i32* %b.y.ptr
  %a.indirect.ptr = getelementptr %struct.point* %a, i32 0, i32 0
  %b.indirect.ptr = getelementptr %struct.point* %b, i32 0, i32 0
  %a.indirect = load i32* %a.indirect.ptr
  %b.indirect = load i32* %b.indirect.ptr
  %0 = add i32 %a.indirect, %b.indirect
  store i32 %0, i32 addrspace(1)* %out
  ret void
}
