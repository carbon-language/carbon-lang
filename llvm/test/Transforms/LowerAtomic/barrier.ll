; RUN: opt < %s -loweratomic -S | FileCheck %s

declare void @llvm.memory.barrier(i1 %ll, i1 %ls, i1 %sl, i1 %ss, i1 %device)

define void @barrier() {
; CHECK: @barrier
  call void @llvm.memory.barrier(i1 0, i1 0, i1 0, i1 0, i1 0)
; CHECK-NEXT: ret
  ret void
}
