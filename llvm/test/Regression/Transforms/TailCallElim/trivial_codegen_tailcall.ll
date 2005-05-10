; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | grep 'tail call void %foo'

declare void %foo()


void %bar() {
	call void %foo()
	ret void
}


