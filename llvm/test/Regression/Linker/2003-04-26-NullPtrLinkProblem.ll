; This one fails because the LLVM runtime is allowing two null pointers of
; the same type to be created!

; RUN: echo "%T = type int" | as > Output/%s.2.bc
; RUN: as < %s > Output/%s.1.bc
; RUN: link Output/%s.[12].bc

%T = type opaque

declare %T* %create()

implementation

void %test() {
	%X = call %T* %create()
	%v = seteq %T* %X, null
	ret void
}

