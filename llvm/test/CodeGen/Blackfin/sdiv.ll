; RUN: llc < %s -march=bfin -verify-machineinstrs
define i32 @sdiv(i32 %A, i32 %B) {
	%R = sdiv i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}
