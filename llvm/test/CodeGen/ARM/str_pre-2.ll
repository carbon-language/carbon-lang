; RUN: llc < %s -mtriple=arm-linux-gnu | FileCheck %s

@b = external global i64*

define i64 @t(i64 %a) nounwind readonly {
entry:
; CHECK: str lr, [sp, #-4]!
; CHECK: ldr lr, [sp], #4
	%0 = load i64** @b, align 4
	%1 = load i64* %0, align 4
	%2 = mul i64 %1, %a
	ret i64 %2
}
