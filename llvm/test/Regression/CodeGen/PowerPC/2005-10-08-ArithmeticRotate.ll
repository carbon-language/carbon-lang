; This was erroneously being turned into an rlwinm instruction.
; The sign bit does matter in this case.

; RUN: llvm-as < %s | llc -march=ppc32 | grep srawi
int %test(int %X) {
	%Y = and int %X, -2
	%Z = shr int %Y, ubyte 11
	ret int %Z
}
