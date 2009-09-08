; RUN: llc < %s -march=x86-64 | FileCheck %s

	%struct.bf = type { i64, i16, i16, i32 }
@bfi = external global %struct.bf*

define void @t1() nounwind ssp {
entry:

; CHECK: andb	$-2, 10(
; CHECK: andb	$-3, 10(

	%0 = load %struct.bf** @bfi, align 8
	%1 = getelementptr %struct.bf* %0, i64 0, i32 1
	%2 = bitcast i16* %1 to i32*
	%3 = load i32* %2, align 1
	%4 = and i32 %3, -65537
	store i32 %4, i32* %2, align 1
	%5 = load %struct.bf** @bfi, align 8
	%6 = getelementptr %struct.bf* %5, i64 0, i32 1
	%7 = bitcast i16* %6 to i32*
	%8 = load i32* %7, align 1
	%9 = and i32 %8, -131073
	store i32 %9, i32* %7, align 1
	ret void
}
