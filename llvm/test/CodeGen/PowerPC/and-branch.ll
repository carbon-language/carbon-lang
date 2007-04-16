; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | not grep mfcr

void %foo(int %X, int %Y, int %Z) {
entry:
	%tmp = seteq int %X, 0		; <bool> [#uses=1]
	%tmp3 = setlt int %Y, 5		; <bool> [#uses=1]
	%tmp4 = and bool %tmp3, %tmp		; <bool> [#uses=1]
	br bool %tmp4, label %cond_true, label %UnifiedReturnBlock

cond_true:		; preds = %entry
	%tmp5 = tail call int (...)* %bar( )		; <int> [#uses=0]
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

declare int %bar(...)
