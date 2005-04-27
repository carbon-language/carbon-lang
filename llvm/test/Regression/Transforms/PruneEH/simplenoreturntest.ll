; RUN: llvm-as < %s | opt -prune-eh | llvm-dis | not grep 'ret int'

void %noreturn() {
	unwind
}

int %caller() {
	 ; noreturn never returns, so the ret is unreachable.
	call void %noreturn()
	ret int 17
}

int %caller2() {
	%T = call int %caller()
	ret int %T            ;; this is also unreachable!
}
