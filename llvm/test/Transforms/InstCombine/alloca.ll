; Zero byte allocas should be deleted.

; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   not grep alloca
; END.

declare void %use(...)
void %test() {
	%X = alloca [0 x int]
	call void(...)* %use([0 x int] *%X)
	%Y = alloca int, uint 0
	call void(...)* %use(int* %Y)
	%Z = alloca {}
	call void(...)* %use({}* %Z)
	ret void
}

void %test2() {
	%A = alloca int    ;; dead.
	store int 123, int* %A
	ret void
}

void %test3() {
	%A = alloca {int}    ;; dead.
	%B = getelementptr {int}* %A, int 0, uint 0
	store int 123, int* %B
	ret void
}
