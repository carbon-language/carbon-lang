; RUN: as < %s | opt -deadargelim | dis | not grep DEADARG

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

internal void %needarg(int %TEST) {
	call int %needarg2(int %TEST)
	ret void
}

internal int %needarg2(int %TEST) {
	ret int %TEST
}

internal void %needarg3(int %TEST3) {
	call void %needarg(int %TEST3)
	ret void
}
