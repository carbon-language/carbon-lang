; RUN: llc < %s -march=x86 | FileCheck %s -check-prefix=X32
; RUN: llc < %s -march=x86-64 | FileCheck %s -check-prefix=X64

; The immediate can be encoded in a smaller way if the
; instruction is a sub instead of an add.

define i32 @test1(i32 inreg %a) nounwind {
  %b = add i32 %a, 128
  ret i32 %b
; X32: subl	$-128, %eax
; X64: subl $-128, 
}
define i64 @test2(i64 inreg %a) nounwind {
  %b = add i64 %a, 2147483648
  ret i64 %b
; X32: addl	$-2147483648, %eax
; X64: subq	$-2147483648,
}
define i64 @test3(i64 inreg %a) nounwind {
  %b = add i64 %a, 128
  ret i64 %b
  
; X32: addl $128, %eax
; X64: subq	$-128,
}
