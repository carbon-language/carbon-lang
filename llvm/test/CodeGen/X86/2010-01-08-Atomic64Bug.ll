; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s
; rdar://r7512579

; PHI defs in the atomic loop should be used by the add / adc
; instructions. They should not be dead.

define void @t(i64* nocapture %p) nounwind ssp {
entry:
; CHECK: t:
; CHECK: movl $1
; CHECK: movl (%ebp), %eax
; CHECK: movl 4(%ebp), %edx
; CHECK: LBB0_1:
; CHECK-NOT: movl $1
; CHECK-NOT: movl $0
; CHECK: addl
; CHECK: adcl
; CHECK: lock
; CHECK: cmpxchg8b
; CHECK: jne
  tail call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  %0 = tail call i64 @llvm.atomic.load.add.i64.p0i64(i64* %p, i64 1) ; <i64> [#uses=0]
  tail call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  ret void
}

declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind

declare i64 @llvm.atomic.load.add.i64.p0i64(i64* nocapture, i64) nounwind
