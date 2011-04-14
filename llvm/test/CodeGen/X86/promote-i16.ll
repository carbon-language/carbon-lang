; RUN: llc < %s -march=x86 | FileCheck %s

define signext i16 @foo(i16 signext %x) nounwind {
entry:
; CHECK: foo:
; CHECK: movzwl 4(%esp), %eax
; CHECK: xorl $21998, %eax
; CHECK: movswl %ax, %eax
  %0 = xor i16 %x, 21998
  ret i16 %0
}
