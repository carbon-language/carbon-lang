; Mem2reg should not insert dead PHI nodes!  The naive algorithm inserts a PHI
;  node in L3, even though there is no load of %A in anything dominated by L3.

; RUN: llvm-as < %s | opt -mem2reg | llvm-dis | not grep phi

void %test(int %B, bool %C) {
	%A = alloca int
	store int %B, int* %A
	br bool %C, label %L1, label %L2
L1:
	store int %B, int* %A
	%D = load int* %A
	call void %test(int %D, bool false)
	br label %L3
L2:
	%E = load int* %A
	call void %test(int %E, bool true)
	br label %L3
L3:
	ret void
}
