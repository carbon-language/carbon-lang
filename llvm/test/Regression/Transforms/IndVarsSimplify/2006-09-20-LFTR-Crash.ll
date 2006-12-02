; RUN: llvm-upgrade < %s | llvm-as | opt -indvars -disable-output

	%struct.p7prior_s = type { int, int, [200 x float], [200 x [7 x float]], int, [200 x float], [200 x [20 x float]], int, [200 x float], [200 x [20 x float]] }

implementation   ; Functions:

void %P7DefaultPrior() {
entry:
	switch int 0, label %UnifiedReturnBlock [
		 int 2, label %bb160
		 int 3, label %bb
	]

bb:		; preds = %entry
	br bool false, label %cond_true.i, label %sre_malloc.exit

cond_true.i:		; preds = %bb
	unreachable

sre_malloc.exit:		; preds = %bb
	br label %cond_true

cond_true:		; preds = %cond_true66, %cond_true, %sre_malloc.exit
	%tmp59 = phi int [ 1, %sre_malloc.exit ], [ %phitmp, %cond_true66 ], [ %tmp59, %cond_true ]		; <int> [#uses=2]
	%indvar245.0.ph = phi uint [ 0, %sre_malloc.exit ], [ %indvar.next246, %cond_true66 ], [ %indvar245.0.ph, %cond_true ]		; <uint> [#uses=2]
	br bool false, label %bb57, label %cond_true

bb57:		; preds = %cond_true
	%tmp65 = setgt int 0, %tmp59		; <bool> [#uses=1]
	%indvar.next246 = add uint %indvar245.0.ph, 1		; <uint> [#uses=2]
	br bool %tmp65, label %cond_true66, label %bb69

cond_true66:		; preds = %bb57
	%q.1.0 = cast uint %indvar.next246 to int		; <int> [#uses=1]
	%phitmp = add int %q.1.0, 1		; <int> [#uses=1]
	br label %cond_true

bb69:		; preds = %bb57
	ret void

bb160:		; preds = %entry
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}
