; RUN: llvm-as < %s | llc -march=arm
void %f(int %a, int %b, int %c, int %d, int %e) {
entry:
	%a_addr = alloca int		; <int*> [#uses=2]
	%b_addr = alloca int		; <int*> [#uses=2]
	%c_addr = alloca int		; <int*> [#uses=2]
	%d_addr = alloca int		; <int*> [#uses=2]
	%e_addr = alloca int		; <int*> [#uses=2]
	store int %a, int* %a_addr
	store int %b, int* %b_addr
	store int %c, int* %c_addr
	store int %d, int* %d_addr
	store int %e, int* %e_addr
	call void %g( int* %a_addr, int* %b_addr, int* %c_addr, int* %d_addr, int* %e_addr )
	ret void
}

declare void %g(int*, int*, int*, int*, int*)
