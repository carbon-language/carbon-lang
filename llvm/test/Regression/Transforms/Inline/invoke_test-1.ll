; Test that we can inline a simple function, turning the calls in it into invoke
; instructions

; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep 'call[^e]'

declare void %might_throw()

implementation

internal void %callee() {
	call void %might_throw()
	ret void
}

; caller returns true if might_throw throws an exception...
int %caller() {
	invoke void %callee() to label %cont except label %exc
cont:
	ret int 0
exc:
	ret int 1
}
