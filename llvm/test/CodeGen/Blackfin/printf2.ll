; RUN: llvm-as < %s | llc -march=bfin
; XFAIL: *
; Assertion failed: (isUsed(Reg) && "Using an undefined register!"),
; function forward, file RegisterScavenging.cpp, line 182.

declare i32 @printf(i8*, ...)

define i32 @main() {
	%1 = call i32 (i8*, ...)* @printf(i8* undef, i1 undef)
	ret i32 0
}
