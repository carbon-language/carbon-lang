; RUN: llvm-as < %s | llc -march=bfin -verify-machineinstrs
define i32 @add(i32 %A, i32 %B) {
	%R = add i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}
