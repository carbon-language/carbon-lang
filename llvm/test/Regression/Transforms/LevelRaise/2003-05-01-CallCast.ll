; RUN: llvm-as < %s | opt -raise | llvm-dis | not grep cast

void %test(...) { ret void }

void %caller() {
	call void (...) *%test()
	ret void
}
