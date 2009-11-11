; RUN: llc < %s -march=x86 -o %t
; RUN: grep "movl	.24601, 12(%ecx)" %t
; RUN: grep "movl	.48, 8(%ecx)" %t
; RUN: grep "movl	.24, 4(%ecx)" %t
; RUN: grep "movl	.12, (%ecx)" %t

%0 = type { i32, i32, i32, i32 }

define internal fastcc %0 @ReturnBigStruct() nounwind readnone {
entry:
  %0 = insertvalue %0 zeroinitializer, i32 12, 0
  %1 = insertvalue %0 %0, i32 24, 1
  %2 = insertvalue %0 %1, i32 48, 2
  %3 = insertvalue %0 %2, i32 24601, 3
  ret %0 %3
}

