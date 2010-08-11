; RUN: llc < %s -march=thumb -mattr=+v6  | FileCheck %s -check-prefix=V6
; RUN: llc < %s -march=thumb -mattr=+v6m | FileCheck %s -check-prefix=V6M

declare void @llvm.memory.barrier( i1 , i1 , i1 , i1 , i1 )

define void @t1() {
; V6: t1:
; V6_TEMPORARILY_DISABLED: blx {{_*}}sync_synchronize

; V6M: t1:
; V6M: dsb
  call void @llvm.memory.barrier( i1 false, i1 false, i1 false, i1 true, i1 true )
  ret void
}

define void @t2() {
; V6: t2:
; V6_TEMPORARILY_DISABLED: blx {{_*}}sync_synchronize

; V6M: t2:
; V6M: dmb
  call void @llvm.memory.barrier( i1 false, i1 false, i1 false, i1 true, i1 false )
  ret void
}
