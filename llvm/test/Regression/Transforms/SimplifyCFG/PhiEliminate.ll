; Test a bunch of cases where the cfg simplification code should
; be able to fold PHI nodes into computation in common cases.  Folding the PHI
; nodes away allows the branches to be eliminated, performing a simple form of
; 'if conversion'.

; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis > Output/%s.xform
; RUN: not grep phi Output/%s.xform && grep ret Output/%s.xform

declare void %use(bool)
declare void %use(int)

void %test(bool %c, int %V) {
	br bool %c, label %T, label %F
T:
	br label %F
F:
	%B1 = phi bool [true, %0], [false, %T]
	%B2 = phi bool [true, %T], [false, %0]
	%I1 = phi int  [1, %T], [0, %0]
	%I2 = phi int  [1, %0], [0, %T]
	%I3 = phi int  [17, %T], [0, %0]
	%I4 = phi int  [17, %T], [5, %0]
	%I5 = phi int  [%V, %T], [0, %0]
	%I6 = phi int  [%V, %0], [0, %T]
	call void %use(bool %B1)
	call void %use(bool %B2)
	call void %use(int  %I1)
	call void %use(int  %I2)
	call void %use(int  %I3)
	call void %use(int  %I4)
	call void %use(int  %I5)
	call void %use(int  %I6)
	ret void
}

