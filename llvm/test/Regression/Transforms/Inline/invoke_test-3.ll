; Test that any rethrown exceptions in an inlined function are automatically
; turned into branches to the invoke destination.

; RUN: as < %s | opt -inline | dis | not grep 'call void %llvm.unwind'

declare void %might_throw()
declare void %llvm.unwind()

implementation

internal int %callee() {
	invoke void %might_throw() to label %cont except label %exc
cont:
	ret int 0
exc:	; This just rethrows the exception!
	call void %llvm.unwind()
	ret int 123  ; DEAD!
}

; caller returns true if might_throw throws an exception... which gets 
; propagated by callee.
int %caller() {
	%X = invoke int %callee() to label %cont 
		except label %Handler
cont:
	ret int %X
Handler:
	; This consumes an exception thrown by might_throw
	ret int 1
}
