; RUN: llvm-upgrade < %s | llvm-as | opt -lowerswitch

void %test() {
	switch uint 0, label %Next []
Next:
	ret void
}
