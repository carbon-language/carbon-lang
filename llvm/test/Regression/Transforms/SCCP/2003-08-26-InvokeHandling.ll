; The PHI cannot be eliminated from this testcase, SCCP is mishandling invoke's!
; RUN: as < %s | opt -sccp | dis | grep phi

declare void %foo()
int %test(bool %cond) {
Entry:
	br bool %cond, label %Inv, label %Cont
Inv:
	invoke void %foo() to label %Ok except label %Cont
Ok:
	br label %Cont
Cont:
	%X = phi int [0, %Entry], [1,%Ok], [0, %Inv]
	ret int %X
}
