; RUN: llc < %s -march=x86 | grep {(%} | count 1

; Don't duplicate the load.

define fastcc i32 @foo(i32* %p) nounwind {
	%t0 = load i32* %p
	%t2 = and i32 %t0, 10
	%t3 = icmp ne i32 %t2, 0
	br i1 %t3, label %bb63, label %bb76

bb63:
	ret i32 %t2

bb76:
	ret i32 0
}
