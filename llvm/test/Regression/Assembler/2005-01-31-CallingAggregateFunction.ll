; RUN: llvm-as %s -o /dev/null 2>&1 | grep "LLVM functions cannot return aggregate types"

void %test() {
	call {} %foo()
	ret void
}

declare {} %foo()
