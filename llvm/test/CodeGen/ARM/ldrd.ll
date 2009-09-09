; RUN: llc < %s -mtriple=armv6-apple-darwin | grep ldrd
; RUN: llc < %s -mtriple=armv5-apple-darwin | not grep ldrd
; RUN: llc < %s -mtriple=armv6-eabi | not grep ldrd
; rdar://r6949835

@b = external global i64*

define i64 @t(i64 %a) nounwind readonly {
entry:
	%0 = load i64** @b, align 4
	%1 = load i64* %0, align 4
	%2 = mul i64 %1, %a
	ret i64 %2
}
