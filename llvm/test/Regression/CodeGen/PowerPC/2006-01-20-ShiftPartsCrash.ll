; RUN: llvm-as < %s | llc

void %iterative_hash_host_wide_int() {
	%zero = alloca int		; <int*> [#uses=2]
	%b = alloca uint		; <uint*> [#uses=1]
	store int 0, int* %zero
	%tmp = load int* %zero		; <int> [#uses=1]
	%tmp5 = cast int %tmp to uint		; <uint> [#uses=1]
	%tmp6 = add uint %tmp5, 32		; <uint> [#uses=1]
	%tmp6 = cast uint %tmp6 to int		; <int> [#uses=1]
	%tmp7 = load long* null		; <long> [#uses=1]
	%tmp6 = cast int %tmp6 to ubyte		; <ubyte> [#uses=1]
	%tmp8 = shr long %tmp7, ubyte %tmp6		; <long> [#uses=1]
	%tmp8 = cast long %tmp8 to uint		; <uint> [#uses=1]
	store uint %tmp8, uint* %b
	unreachable
}
