; Test that if an invoked function is inlined, and if that function cannot
; throw, that the dead handler is now unreachable.

; RUN: llvm-as < %s | opt -inline -simplifycfg | llvm-dis | not grep UnreachableExceptionHandler

declare void %might_throw()

implementation

internal int %callee() {
	invoke void %might_throw() to label %cont except label %exc
cont:
	ret int 0
exc:
	; This just consumes the exception!
	ret int 1
}

; caller returns true if might_throw throws an exception... callee cannot throw.
int %caller() {
	%X = invoke int %callee() to label %cont 
		except label %UnreachableExceptionHandler
cont:
	ret int %X
UnreachableExceptionHandler:
	ret int -1   ; This is dead!
}
