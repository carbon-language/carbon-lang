; RUN: llc -filetype=obj < %s | llvm-objdump -d - | FileCheck %s

; Test that the direct object emission selects the and variant with 8 bit
; immediate.
; We used to get this wrong when using direct object emission, but not when
; reading assembly.

; CHECK: 48 83 e4 e0                    andq      $-32, %rsp

target triple = "x86_64-pc-linux"

define void @f() {
  %foo = alloca i8, align 32
  ret void
}
