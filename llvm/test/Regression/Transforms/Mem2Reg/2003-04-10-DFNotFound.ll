; RUN: llvm-as < %s | opt -mem2reg

implementation   ; Functions:

void %_Z3barv() {
	%result = alloca int
	ret void

	store int 0, int* %result  ; DF not set!
	ret void
}
