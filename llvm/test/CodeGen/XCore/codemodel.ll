
; RUN: llc < %s -march=xcore | FileCheck %s

; CHECK: .section  .cp.rodata.cst4,"aMc",@progbits,4
; CHECK: .long 65536
; CHECK: .text
; CHECK-LABEL: f:
; CHECK: ldc r1, 65532
; CHECK: add r1, r0, r1
; CHECK: ldw r1, r1[0]
; CHECK: ldw r2, cp[.LCPI0_0]
; CHECK: add r0, r0, r2
; CHECK: ldw r0, r0[0]
; CHECK: add r0, r1, r0
; CHECK: ldw r1, dp[l]
; CHECK: add r0, r0, r1
; CHECK: ldw r1, dp[l+4]
; CHECK: add r0, r0, r1
; CHECK: ldw r1, dp[l+392]
; CHECK: add r0, r0, r1
; CHECK: ldw r1, dp[l+396]
; CHECK: add r0, r0, r1
; CHECK: ldw r1, dp[s]
; CHECK: add r0, r0, r1
; CHECK: ldw r1, dp[s+36]
; CHECK: add r0, r0, r1
; CHECK: retsp 0
define i32 @f(i32* %i) {
entry:
  %0 = getelementptr inbounds i32* %i, i32 16383
  %1 = load i32* %0
  %2 = getelementptr inbounds i32* %i, i32 16384
  %3 = load i32* %2
  %4 = add nsw i32 %1, %3
  %5 = load i32* getelementptr inbounds ([100 x i32]* @l, i32 0, i32 0)
  %6 = add nsw i32 %4, %5
  %7 = load i32* getelementptr inbounds ([100 x i32]* @l, i32 0, i32 1)
  %8 = add nsw i32 %6, %7
  %9 = load i32* getelementptr inbounds ([100 x i32]* @l, i32 0, i32 98)
  %10 = add nsw i32 %8, %9
  %11 = load i32* getelementptr inbounds ([100 x i32]* @l, i32 0, i32 99)
  %12 = add nsw i32 %10, %11
  %13 = load i32* getelementptr inbounds ([10 x i32]* @s, i32 0, i32 0)
  %14 = add nsw i32 %12, %13
  %15 = load i32* getelementptr inbounds ([10 x i32]* @s, i32 0, i32 9)
  %16 = add nsw i32 %14, %15
  ret i32 %16
}

; CHECK: .section .dp.bss,"awd",@nobits
; CHECK-LABEL: l:
; CHECK: .space 400
@l = global [100 x i32] zeroinitializer

; CHECK-LABEL: s:
; CHECK: .space 40
@s = global [10 x i32] zeroinitializer

; CHECK: .section .cp.rodata,"ac",@progbits
; CHECK-LABEL: cl:
; CHECK: .space 400
@cl = constant  [100 x i32] zeroinitializer

; CHECK-LABEL: cs:
; CHECK: .space 40
@cs = constant  [10 x i32] zeroinitializer
