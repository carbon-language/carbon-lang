; Test that functions with dynamic allocas get inlined in a case where
; naively inlining it would result in a miscompilation.

; RUN: llvm-as < %s | opt -inline &&
; RUN: llvm-as < %s | opt -inline | llvm-dis | grep llvm.stacksave &&
; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep callee

declare void %ext(int*)
implementation

internal void %callee(uint %N) {
	%P = alloca int, uint %N     ;; dynamic alloca
	call void %ext(int* %P)
	ret void
}

void %foo(uint %N) {
	br label %Loop
Loop:
	%count = phi uint [0, %0], [%next, %Loop]
	%next = add uint %count, 1
	call void %callee(uint %N)
	%cond = seteq uint %count, 100000
	br bool %cond, label %out, label %Loop
out:
	ret void
}

