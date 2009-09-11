; RUN: opt < %s -ipsccp -S > %t
; RUN: grep {ret i32 36} %t
; RUN: grep {%mrv = insertvalue %T undef, i32 18, 0} %t
; RUN: grep {%mrv1 = insertvalue %T %mrv, i32 17, 1} %t
; RUN: grep {ret %T %mrv1} %t

%T = type {i32,i32}

define internal {i32, i32} @bar(i32 %A) {
	%X = add i32 1, %A
	ret i32 %X, i32 %A
}

define i32 @foo() {
	%X = call {i32, i32} @bar(i32 17)
        %Y = getresult {i32, i32} %X, 0
	%Z = add i32 %Y, %Y
	ret i32 %Z
}
