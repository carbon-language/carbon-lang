; RUN: llvm-as < %s | llc -march=x86 | grep movl | wc -l | grep 1

%dst = global int 0
%ptr = global int* null

void %test() {
	store int* %dst, int** %ptr
	ret void
}
