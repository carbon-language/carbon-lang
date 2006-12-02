; RUN: llvm-upgrade %s | llvm-as -o /dev/null -f 2>&1 | grep "LLVM functions cannot return aggregate types"

void %test() {
	call {} %foo()
	ret void
}

declare {} %foo()
