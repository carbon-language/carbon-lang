; RUN: llc %s -o - -mtriple=x86_64-linux-gnu | FileCheck %s

; This can get rounded up to the preferred alignment (16).
; PR6921
@GlobalA = global { [384 x i8] } zeroinitializer, align 8 

; CHECK:	.bss
; CHECK:	.globl	GlobalA
; CHECK:	.align	16
; CHECK: GlobalA:
; CHECK:	.zero	384

; Common variables should also get rounded up to the preferred alignment (16).
@GlobalB = common global { [384 x i8] } zeroinitializer, align 8

; CHECK: 	.comm	GlobalB,384,16 