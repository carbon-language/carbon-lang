; RUN: as < %s | opt -lowerswitch

void %test() {
	switch uint 0, label %Next []
Next:
	ret void
}
