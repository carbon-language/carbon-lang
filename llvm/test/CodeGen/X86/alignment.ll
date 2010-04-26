; RUN: llc %s -o - -mtriple=x86_64-linux-gnu | FileCheck %s

; This cannot get rounded up to the preferred alignment (16) if they have an
; explicit alignment specified.
@GlobalA = global { [384 x i8] } zeroinitializer, align 8 

; CHECK:	.bss
; CHECK:	.globl	GlobalA
; CHECK:	.align	8
; CHECK: GlobalA:
; CHECK:	.zero	384

; Common variables should not get rounded up to the preferred alignment (16) if
; they have an explicit alignment specified.
; PR6921
@GlobalB = common global { [384 x i8] } zeroinitializer, align 8

; CHECK: 	.comm	GlobalB,384,8