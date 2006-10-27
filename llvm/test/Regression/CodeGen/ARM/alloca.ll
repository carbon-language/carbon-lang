; RUN: llvm-as < %s | llc -march=arm
void %f(uint %a) {
entry:
	%tmp1032 = alloca ubyte, uint %a
	ret void
}
