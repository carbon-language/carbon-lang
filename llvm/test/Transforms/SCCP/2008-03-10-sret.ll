; RUN: llvm-as < %s  | opt -ipsccp -disable-output

define internal {i32, i32} @bar(i32 %A) {
	%X = add i32 1, 2
	ret i32 %A, i32 %A
}

define i32 @foo() {
	%X = call {i32, i32} @bar(i32 17)
        %Y = getresult {i32, i32} %X, 0
	ret i32 %Y
}
