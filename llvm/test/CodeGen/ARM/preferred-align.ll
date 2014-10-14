; RUN: llc -mtriple=armv7-linux-gnueabi %s -o - | FileCheck %s

@var_agg = global {i8, i8} zeroinitializer

; CHECK: .globl var_agg
; CHECK-NEXT: .align 2

@var1 = global i1 zeroinitializer

; CHECK: .globl var1
; CHECK-NOT: .align

@var8 = global i8 zeroinitializer

; CHECK: .globl var8
; CHECK-NOT: .align

@var16 = global i16 zeroinitializer

; CHECK: .globl var16
; CHECK-NEXT: .align 1