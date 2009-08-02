; RUN: llvm-as < %s | llc -march=bfin > %t
; XFAIL: *
; Assertion failed: (isUsed(Reg) && "Using an undefined register!")
; function forward, file RegisterScavenging.cpp, line 259.

define i1 @cmp3(i32 %A) {
	%R = icmp uge i32 %A, 2
	ret i1 %R
}
