; RUN: llc < %s -march=thumb -mattr=+v6      | FileCheck %s -check-prefix=V6
; RUN: llc < %s -march=thumb -mcpu=cortex-m0 | FileCheck %s -check-prefix=M0

declare void @llvm.memory.barrier( i1 , i1 , i1 , i1 , i1 )

define void @t1() {
; V6: t1:
; V6: blx {{_*}}sync_synchronize

; M0: t1:
; M0: dsb
  call void @llvm.memory.barrier( i1 false, i1 false, i1 false, i1 true, i1 true )
  ret void
}

define void @t2() {
; V6: t2:
; V6: blx {{_*}}sync_synchronize

; M0: t2:
; M0: dmb
  call void @llvm.memory.barrier( i1 false, i1 false, i1 false, i1 true, i1 false )
  ret void
}
