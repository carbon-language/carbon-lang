; RUN: llvm-as < %s | opt -inline -disable-output

; Inlining the first call caused the inliner function to delete the second
; call.  Then the inliner tries to inline the second call, which no longer
; exists.

implementation

internal void %Callee1() {
	unwind
}

void %Callee2() {
	ret void
}

void %caller() {
	call void %Callee1()
	call void %Callee2()
	ret void
}
