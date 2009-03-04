; RUN: llvm-as < %s | llc -march=x86    | grep call | grep 12345678
; RUN: llvm-as < %s | llc -march=x86-64 | grep call | grep 12345678
; PR3666

define i32 @main() nounwind {
entry:
	%0 = call i32 inttoptr (i32 12345678 to i32 (i32)*)(i32 0) nounwind		; <i32> [#uses=1]
	ret i32 %0
}
