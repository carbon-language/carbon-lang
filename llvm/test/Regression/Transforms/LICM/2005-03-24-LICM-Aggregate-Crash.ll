; RUN: llvm-as < %s | opt -licm -disable-output

void %test({int}* %P) {
	br label %Loop

Loop:
	free {int}* %P
	br label %Loop
}

