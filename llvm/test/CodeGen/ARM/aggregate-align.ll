; RUN: llc -mtriple=armv7-linux-gnueabi %s -o - | FileCheck %s

@var = global {i8, i8} zeroinitializer

; CHECK: .globl var
; CHECK-NEXT: .align 2
