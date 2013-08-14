; RUN: llc < %s -mcpu=generic -march=x86-64 -asm-verbose=false | FileCheck %s

define zeroext i1 @test0(i64 %x) nounwind {
; CHECK-LABEL: test0:
; CHECK-NEXT: incq %[[X:rdi|rcx]]
; CHECK-NEXT: cmpq $1, %[[X]]
; CHECK-NEXT: seta %al
; CHECK-NEXT: ret
  %cmp1 = icmp ne i64 %x, -1
  %not.cmp = icmp ne i64 %x, 0
  %.cmp1 = and i1 %cmp1, %not.cmp
  ret i1 %.cmp1
}
