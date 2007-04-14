; RUN: llvm-upgrade < %s | llvm-as | opt -basicaa -load-vn -gcse -instcombine |\
; RUN:   llvm-dis | grep {ret i32 0}
declare void %foo(int*)
declare void %bar()

int %test() {
	%A = alloca int
	call void %foo(int* %A)

	%X = load int* %A
	tail call void %bar()   ;; Cannot modify *%A because it's on the stack.
	%Y = load int* %A
	%Z = sub int %X, %Y
	ret int %Z
}


