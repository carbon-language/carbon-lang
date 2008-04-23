; RUN: llvm-as < %s  | opt -ipsccp | llvm-dis | grep {ret i32 36}
; RUN: llvm-as < %s  | opt -ipsccp | llvm-dis | grep {ret i32 18, i32 17}

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
