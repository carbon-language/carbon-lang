; RUN: llc < %s -march=bfin > %t

define i1 @cmp3(i32 %A) {
	%R = icmp uge i32 %A, 2
	ret i1 %R
}
