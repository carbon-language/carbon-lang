; Test that any rethrown exceptions in an inlined function are automatically
; turned into branches to the invoke destination.

; RUN: as < %s | opt -inline | dis | not grep 'call void %llvm.exc.rethrow'

declare void %might_throw()
declare void %llvm.exc.rethrow()

implementation

internal int %callee() {
	invoke void %might_throw() to label %cont except label %exc
cont:
	ret int 0
exc:	; This just rethrows the exception!
	call void %llvm.exc.rethrow()
	ret int 0
}

; caller returns true if might_throw throws an exception...
int %caller() {
	%X = invoke int %callee() to label %cont 
		except label %Handler
cont:
	ret int %X
Handler:
	; This consumes an exception thrown by might_throw
	ret int -1
}
