; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep br

int %test1(bool %C) {
entry:
        br bool %C, label %T, label %F
T:
        ret int 1
F:
        ret int 0
}

void %test2(bool %C) {
	br bool %C, label %T, label %F
T:
	ret void
F:
	ret void
}
