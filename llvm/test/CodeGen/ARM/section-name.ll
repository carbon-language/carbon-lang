; RUN: llc < %s -mtriple=arm-unknown-linux-gnueabi | FileCheck %s

; CHECK: .text
; CHECK: .globl test1
; CHECK: .type test1,%function
define void @test1() {
entry:
  ret void
}

; CHECK: .section .test2,"ax",%progbits
; CHECK: .globl test2
; CHECK: .type test2,%function
define void @test2() section ".test2" {
entry:
  ret void
}

; CHECK: .section .text.test3,"axG",%progbits,test3,comdat
; CHECK: .weak test3
; CHECK: .type test3,%function
define linkonce_odr void @test3() {
entry:
  ret void
}
