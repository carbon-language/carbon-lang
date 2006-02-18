; RUN: llvm-as < %s | opt -simplifycfg -disable-output

void %polnel_() {
entry:
	%tmp595 = setlt int 0, 0		; <bool> [#uses=4]
	br bool %tmp595, label %bb148.critedge, label %cond_true40

bb36:		; preds = %bb43
	br bool %tmp595, label %bb43, label %cond_true40

cond_true40:		; preds = %bb46, %cond_true40, %bb36, %entry
	%tmp397 = setgt int 0, 0		; <bool> [#uses=1]
	br bool %tmp397, label %bb43, label %cond_true40

bb43:		; preds = %cond_true40, %bb36
	br bool false, label %bb53, label %bb36

bb46:		; preds = %bb53
	br bool %tmp595, label %bb53, label %cond_true40

bb53:		; preds = %bb46, %bb43
	br bool false, label %bb102, label %bb46

bb92.preheader:		; preds = %bb102
	ret void

bb102:		; preds = %bb53
	br bool %tmp595, label %bb148, label %bb92.preheader

bb148.critedge:		; preds = %entry
	ret void

bb148:		; preds = %bb102
	ret void
}
