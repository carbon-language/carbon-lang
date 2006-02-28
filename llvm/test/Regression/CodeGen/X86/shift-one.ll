; RUN: llvm-as < %s | llc -march=x86 | not grep 'leal'

%x = external global int

int %test() {
	%tmp.0 = load int* %x
	%tmp.1 = shl int %tmp.0, ubyte 1
	ret int %tmp.1
}
