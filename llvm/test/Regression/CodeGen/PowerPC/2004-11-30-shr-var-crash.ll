; RUN: llvm-as < %s | llc -march=ppc32

void %main() {
	%shamt = add ubyte 0, 1		; <ubyte> [#uses=1]
	%tr2 = shr long 1, ubyte %shamt		; <long> [#uses=0]
	ret void
}
