; RUN: llc < %s -march=x86 | FileCheck %s
; <rdar://problem/8449754>

define i32 @add32carry(i32 %sum, i32 %x) nounwind readnone ssp {
entry:
; CHECK:	sbbl	%ecx, %ecx
; CHECK-NOT: addl
; CHECK: subl	%ecx, %eax
  %add4 = add i32 %x, %sum
  %cmp = icmp ult i32 %add4, %x
  %inc = zext i1 %cmp to i32
  %z.0 = add i32 %add4, %inc
  ret i32 %z.0
}
