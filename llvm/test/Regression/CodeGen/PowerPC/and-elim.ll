; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep rlwin

void %test(ubyte* %P) {
	%W = load ubyte* %P
	%X = shl ubyte %W, ubyte 1
	%Y = add ubyte %X, 2
	%Z = and ubyte %Y, 254        ; dead and
	store ubyte %Z, ubyte* %P
	ret void
}
