; RUN: llvm-as < %s | opt -raise | llvm-dis | notcast

void %test(...) { ret void }

void %caller() {
	call void (...) *%test()
	ret void
}
