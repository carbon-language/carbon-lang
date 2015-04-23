; RUN: llc -relocation-model=pic -mtriple=thumbv7-unknown-linux -o - %s | FileCheck %s

@x = external global i32

; CHECK: .globl	foo
; CHECK-NEXT: .align	2
define i32* @foo() {
  ret i32* @x
}

; CHECK: .globl	bar
; CHECK-NEXT: .align	1
define i32* @bar() {
  ret i32* zeroinitializer
}
