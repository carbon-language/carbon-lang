; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -disable-output

%p = weak global int 0

int %test(int %x) {
	%y = mul int %x, cast (int* %p to int)
	ret int %y
}
