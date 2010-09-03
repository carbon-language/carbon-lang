; RUN: llc < %s -mtriple=i686-linux | not egrep {\\\$4294967289|-7}
; RUN: llc < %s -mtriple=i686-linux | egrep {\\\$4294967280|-16}
; RUN: llc < %s -mtriple=x86_64-linux | grep {\\-16}

define void @t() nounwind {
A:
	br label %entry

entry:
	%m1 = alloca i32, align 4
	%m2 = alloca [7 x i8], align 16
	call void @s( i32* %m1, [7 x i8]* %m2 )
	ret void
}

declare void @s(i32*, [7 x i8]*)
