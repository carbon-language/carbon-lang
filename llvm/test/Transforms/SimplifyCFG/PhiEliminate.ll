; Test a bunch of cases where the cfg simplification code should
; be able to fold PHI nodes into computation in common cases.  Folding the PHI
; nodes away allows the branches to be eliminated, performing a simple form of
; 'if conversion'.

; RUN: llvm-upgrade < %s | llvm-as | opt -simplifycfg | llvm-dis > %t.xform
; RUN:   not grep phi %t.xform 
; RUN:   grep ret %t.xform

declare void %use(bool)
declare void %use(int)


void %test2(bool %c, bool %d, int %V, int %V2) {
	br bool %d, label %X, label %F
X:
	br bool %c, label %T, label %F
T:
	br label %F
F:
	%B1 = phi bool [true, %0], [false, %T], [false, %X]
	%I7 = phi int  [%V, %0], [%V2, %T], [%V2, %X]
	call void %use(bool %B1)
	call void %use(int  %I7)
	ret void
}

void %test(bool %c, int %V, int %V2) {
	br bool %c, label %T, label %F
T:
	br label %F
F:
	%B1 = phi bool [true, %0], [false, %T]
	%I6 = phi int  [%V, %0], [0, %T]
	call void %use(bool %B1)
	call void %use(int  %I6)
	ret void
}
