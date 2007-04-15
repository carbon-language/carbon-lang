; RUN: llvm-upgrade < %s | llvm-as | opt -tailcallelim | llvm-dis | \
; RUN:    grep {tail call void @foo}

declare void %foo()


void %bar() {
	call void %foo()
	ret void
}


