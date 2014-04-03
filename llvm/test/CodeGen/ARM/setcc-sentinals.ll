; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 -asm-verbose=false %s -o - | FileCheck %s

define zeroext i1 @test0(i32 %x) nounwind {
; CHECK-LABEL: test0:
; CHECK: add [[REG:(r[0-9]+)|(lr)]], r0, #1
; CHECK-NEXT: mov r0, #0
; CHECK-NEXT: cmp [[REG]], #1
; CHECK-NEXT: movwhi r0, #1
; CHECK-NEXT: bx  lr
  %cmp1 = icmp ne i32 %x, -1
  %not.cmp = icmp ne i32 %x, 0
  %.cmp1 = and i1 %cmp1, %not.cmp
  ret i1 %.cmp1
}
