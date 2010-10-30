; RUN: llc < %s -march=thumb -mcpu=cortex-a8 | FileCheck %s

declare void @llvm.memory.barrier(i1 , i1 , i1 , i1 , i1)

define void @t_st() {
; CHECK: t_st:
; CHECK: dmb st
  call void @llvm.memory.barrier(i1 false, i1 false, i1 false, i1 true, i1 true)
  ret void
}

define void @t_sy() {
; CHECK: t_sy:
; CHECK: dmb sy
  call void @llvm.memory.barrier(i1 true, i1 false, i1 false, i1 true, i1 true)
  ret void
}

define void @t_ishst() {
; CHECK: t_ishst:
; CHECK: dmb ishst
  call void @llvm.memory.barrier(i1 false, i1 false, i1 false, i1 true, i1 false)
  ret void
}

define void @t_ish() {
; CHECK: t_ish:
; CHECK: dmb ish
  call void @llvm.memory.barrier(i1 true, i1 false, i1 false, i1 true, i1 false)
  ret void
}
