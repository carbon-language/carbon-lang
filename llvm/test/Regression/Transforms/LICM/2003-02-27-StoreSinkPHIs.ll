; LICM is adding stores before phi nodes.  bad.

; RUN: llvm-as < %s | opt -licm

bool %test(bool %c) {
	br bool %c, label %Loop, label %Out
Loop:
	store int 0, int* null
	br bool %c, label %Loop, label %Out
Out:
	%X = phi bool [%c, %0], [true, %Loop]
	ret bool %X
}
