; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -mem2reg | llvm-dis | \
; RUN:    not grep load

int %test1(uint* %P) {
	%A = alloca uint 
	store uint 123, uint* %A
	%Q = cast uint* %A to int*    ; Cast the result of the load not the source
	%V = load int* %Q
	ret int %V
}
