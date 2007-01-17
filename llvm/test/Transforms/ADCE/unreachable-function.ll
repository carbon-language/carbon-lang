; RUN: llvm-upgrade < %s | llvm-as | opt -adce -disable-output

void %test() {
	unreachable
}
