; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6,+vfp2 | \
; RUN:   grep fcmpes

void %test3(float* %glob, int %X) {
entry:
	%tmp = load float* %glob		; <float> [#uses=1]
	%tmp2 = getelementptr float* %glob, int 2		; <float*> [#uses=1]
	%tmp3 = load float* %tmp2		; <float> [#uses=1]
	%tmp = setgt float %tmp, %tmp3		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %UnifiedReturnBlock

cond_true:		; preds = %entry
	%tmp = tail call int (...)* %bar( )		; <int> [#uses=0]
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

declare int %bar(...)
