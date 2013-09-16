; RUN: echo > %t.ll
; RUN: llvm-link %t.ll %s -S -o - | FileCheck %s

@i = linkonce_odr global i32 1

; CHECK: define void @f() prefix i32* @i
define void @f() prefix i32* @i {
  ret void
}
