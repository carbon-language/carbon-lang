; RUN: llvm-as < %s | opt -no-aa -ds-aa -load-vn -gcse | llvm-dis | not grep load
%T = type { int*, int* }

int %main() {
	%A = alloca %T
	%B = alloca { %T }
	%C = alloca %T*
	%Bp = getelementptr { %T }* %B, long 0, ubyte 0

	%i0 = alloca int
	%i1 = alloca int
	%Ap0 = getelementptr %T* %A, long 0, ubyte 0
	%Ap1 = getelementptr %T* %A, long 0, ubyte 1
	store int* %i0, int** %Ap0
	store int* %i1, int** %Ap1

	store int 0, int* %i0
	store int 1, int* %i1
	%RetVal = load int* %i0   ; This load should be deletable

	store %T* %A, %T** %C
	store %T* %Bp, %T** %C    ; This store was causing merging to happen!
	ret int %RetVal
}
