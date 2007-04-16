; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32
; RUN: llvm-upgrade < %s | llvm-as | llc

%qsz.b = external global bool		; <bool*> [#uses=1]

implementation   ; Functions:

fastcc void %qst() {
entry:
	br bool true, label %cond_next71, label %cond_true

cond_true:		; preds = %entry
	ret void

cond_next71:		; preds = %entry
	%tmp73.b = load bool* %qsz.b		; <bool> [#uses=1]
	%ii.4.ph = select bool %tmp73.b, ulong 4, ulong 0		; <ulong> [#uses=1]
	br label %bb139

bb82:		; preds = %bb139
	ret void

bb139:		; preds = %bb139, %cond_next71
	%exitcond89 = seteq ulong 0, %ii.4.ph		; <bool> [#uses=1]
	br bool %exitcond89, label %bb82, label %bb139
}
