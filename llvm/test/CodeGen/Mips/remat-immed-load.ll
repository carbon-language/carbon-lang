; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=32
; RUN: llc -march=mips64el -mcpu=mips64 -mattr=n64 < %s | FileCheck %s -check-prefix=64

define void @f0() nounwind {
entry:
; 32:  addiu $4, $zero, 1
; 32:  addiu $4, $zero, 1

  tail call void @foo1(i32 1) nounwind
  tail call void @foo1(i32 1) nounwind
  ret void
}

declare void @foo1(i32)

define void @f3() nounwind {
entry:
; 64:  daddiu $4, $zero, 1
; 64:  daddiu $4, $zero, 1

  tail call void @foo2(i64 1) nounwind
  tail call void @foo2(i64 1) nounwind
  ret void
}

declare void @foo2(i64)
