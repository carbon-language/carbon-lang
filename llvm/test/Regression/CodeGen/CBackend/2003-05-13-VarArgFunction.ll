; This testcase breaks the C backend, because gcc doesn't like (...) functions
; with no arguments at all.

void %test(long %Ptr) {
	%P = cast long %Ptr to void(...) *
	call void(...)* %P(long %Ptr)
	ret void
}
