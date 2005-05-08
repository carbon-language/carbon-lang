; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | not grep sub

; Test that we can turn things like X*-(Y*Z) -> X*-1*Y*Z.

int %test1(int %a, int %b, int %z) {
	%c = sub int 0, %z
	%d = mul int %a, %b
	%e = mul int %c, %d
	%f = mul int %e, 12345
	%g = sub int 0, %f
	ret int %g
}

int %test2(int %a, int %b, int %z) {
	%d = mul int %z, 40
	%c = sub int 0, %d
	%e = mul int %a, %c
	%f = sub int 0, %e
	ret int %f
}
