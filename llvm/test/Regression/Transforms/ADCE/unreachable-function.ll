; RUN: llvm-as < %s | opt -adce -disable-output

void %test() {
	unreachable
}
