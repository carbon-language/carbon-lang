; LoopSimplify is breaking LICM on this testcase because the exit blocks from
; the loop are reachable from more than just the exit nodes: the exit blocks
; have predecessors from outside of the loop!
;
; This is distilled from a monsterous crafty example.

; RUN: llvm-as < %s | opt -licm | lli

%G = weak global int 0		; <int*> [#uses=13]

implementation   ; Functions:

int %main() {
entry:
	store int 123, int* %G
	br label %loopentry.i

loopentry.i:		; preds = %entry, %endif.1.i
	%tmp.0.i = load int* %G		; <int> [#uses=1]
	%tmp.1.i = seteq int %tmp.0.i, 123		; <bool> [#uses=1]
	br bool %tmp.1.i, label %Out.i, label %endif.0.i

endif.0.i:		; preds = %loopentry.i
	%tmp.3.i = load int* %G		; <int> [#uses=1]
	%tmp.4.i = seteq int %tmp.3.i, 126		; <bool> [#uses=1]
	br bool %tmp.4.i, label %ExitBlock.i, label %endif.1.i

endif.1.i:		; preds = %endif.0.i
	%tmp.6.i = load int* %G		; <int> [#uses=1]
	%inc.i = add int %tmp.6.i, 1		; <int> [#uses=1]
	store int %inc.i, int* %G
	br label %loopentry.i

Out.i:		; preds = %loopentry.i
	store int 0, int* %G
	br label %ExitBlock.i

ExitBlock.i:		; preds = %endif.0.i, %Out.i
	%tmp.7.i = load int* %G		; <int> [#uses=1]
	ret int %tmp.7.i
}

