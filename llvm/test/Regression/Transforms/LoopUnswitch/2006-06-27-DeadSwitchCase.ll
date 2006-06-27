; RUN: llvm-as < %s | opt -loop-unswitch -disable-output
implementation   ; Functions:

void %init_caller_save() {
entry:
	br label %cond_true78

cond_next20:		; preds = %cond_true64
	br label %bb31

bb31:		; preds = %cond_true64, %cond_true64, %cond_next20
	%iftmp.29.1 = phi uint [ 0, %cond_next20 ], [ 0, %cond_true64 ], [ 0, %cond_true64 ]		; <uint> [#uses=0]
	br label %bb54

bb54:		; preds = %cond_true78, %bb31
	br bool false, label %bb75, label %cond_true64

cond_true64:		; preds = %bb54
	switch int %i.0.0, label %cond_next20 [
		 int 17, label %bb31
		 int 18, label %bb31
	]

bb75:		; preds = %bb54
	%tmp74.0 = add int %i.0.0, 1		; <int> [#uses=1]
	br label %cond_true78

cond_true78:		; preds = %bb75, %entry
	%i.0.0 = phi int [ 0, %entry ], [ %tmp74.0, %bb75 ]		; <int> [#uses=2]
	br label %bb54
}
