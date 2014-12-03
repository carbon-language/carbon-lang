; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

@i = linkonce_odr global i32 1

; CHECK: .type f,@function
; CHECK-NEXT: .long	1
; CHECK-NEXT: # 0x1
; CHECK-NEXT: f:
define void @f() prefix i32 1 {
  ret void
}

; CHECK: .type g,@function
; CHECK-NEXT: .quad	i
; CHECK-NEXT: g:
define void @g() prefix i32* @i {
  ret void
}
