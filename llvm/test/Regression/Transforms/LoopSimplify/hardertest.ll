; RUN: llvm-as < %s | opt -preheaders

void %foo(bool %C) {
	br bool %C, label %T, label %F
T:
	br label %Loop
F: 
	br label %Loop

Loop:    ; Two backedges, two incoming edges.
	%Val = phi int [0, %T], [1, %F], [2, %Loop], [3, %L2]

	br bool %C, label %Loop, label %L2

L2:
	br label %Loop
}
