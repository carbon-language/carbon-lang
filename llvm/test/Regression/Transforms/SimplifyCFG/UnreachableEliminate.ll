; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep unreachable

void %test1(bool %C, bool* %BP) {
	br bool %C, label %T, label %F
T:
	store bool %C, bool* %BP  ;; dead
	unreachable
F:
	ret void
}

void %test2() {
	invoke void %test2() to label %N unwind label %U
U:
	unreachable
N:
	ret void
}

int %test3(int %v) {
	switch int %v, label %default [ int 1, label %U
                                        int 2, label %T]
default:
	ret int 1
U:
	unreachable
T:
	ret int 2
}
