;RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep 'br label'
void %test(bool %C) {
	br bool %C, label %A, label %B
A:
	call void %test(bool %C)
	br label %X
B:
	call void %test(bool %C)
	br label %X
X:
	unwind
}
