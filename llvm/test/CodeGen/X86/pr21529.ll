; RUN: llc -show-mc-encoding < %s | FileCheck %s

; Test that the direct object emission selects the and variant with 8 bit
; immediate.
; We used to get this wrong when using direct object emission, but not when
; reading assembly.

; CHECK: andq    $-32, %rsp              # encoding: [0x48,0x83,0xe4,0xe0]

target triple = "x86_64-pc-linux"

define void @f() {
  %foo = alloca i8, align 32
  ret void
}
