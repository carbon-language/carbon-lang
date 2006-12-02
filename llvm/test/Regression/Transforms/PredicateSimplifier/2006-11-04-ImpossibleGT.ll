; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify -disable-output

void %readMotionInfoFromNAL() {
entry:
	br bool false, label %bb2425, label %cond_next30

cond_next30:		; preds = %entry
	ret void

bb2418:		; preds = %bb2425
	ret void

bb2425:		; preds = %entry
	%tmp2427 = setgt int 0, 3		; <bool> [#uses=1]
	br bool %tmp2427, label %cond_next2429, label %bb2418

cond_next2429:		; preds = %bb2425
	ret void
}
