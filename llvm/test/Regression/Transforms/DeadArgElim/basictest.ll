; RUN: if as < %s | opt -load ~/llvm/lib/Debug/libhello.so -deadargelim | dis | grep DEADARG
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

; test - an obviously dead argument
internal int %test(int %v, int %DEADARG1, int* %p) {
	store int %v, int* %p
	ret int %v
}

; hardertest - an argument which is only used by a call of a function with a 
; dead argument.
internal int %hardertest(int %DEADARG2) {
	%p = alloca int
	%V = call int %test(int 5, int %DEADARG2, int* %p)
	ret int %V
}

; evenhardertest - recursive dead argument...
internal void %evenhardertest(int %DEADARG3) {
	call void %evenhardertest(int %DEADARG3)
	ret void
}

