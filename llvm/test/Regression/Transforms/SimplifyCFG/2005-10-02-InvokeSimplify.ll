; RUN: llvm-as < %s | opt -simplifycfg -disable-output

bool %foo() {
	%X = invoke bool %foo() to label %N unwind label %F
F:
	ret bool false
N:
	br bool %X, label %A, label %B
A:
	ret bool true
B:
	ret bool true
}
