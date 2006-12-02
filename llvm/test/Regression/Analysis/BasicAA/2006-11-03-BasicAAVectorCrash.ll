; RUN: llvm-upgrade < %s | llvm-as | opt -licm -disable-output
target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8.7.0"

implementation   ; Functions:

void %glgRunProcessor() {
entry:
	br bool false, label %bb2037.i, label %cond_true.i18

cond_true.i18:		; preds = %entry
	ret void

bb205.i:		; preds = %bb2037.i
	switch uint 0, label %bb1013.i [
		 uint 14, label %bb239.i
		 uint 15, label %bb917.i
	]

bb239.i:		; preds = %bb205.i
	br bool false, label %cond_false277.i, label %cond_true264.i

cond_true264.i:		; preds = %bb239.i
	ret void

cond_false277.i:		; preds = %bb239.i
	%tmp1062.i = getelementptr [2 x <4 x int>]* null, int 0, int 1		; <<4 x int>*> [#uses=1]
	store <4 x int> zeroinitializer, <4 x int>* %tmp1062.i
	br bool false, label %cond_true1032.i, label %cond_false1063.i85

bb917.i:		; preds = %bb205.i
	ret void

bb1013.i:		; preds = %bb205.i
	ret void

cond_true1032.i:		; preds = %cond_false277.i
	%tmp1187.i = getelementptr [2 x <4 x int>]* null, int 0, int 0, int 7		; <int*> [#uses=1]
	store int 0, int* %tmp1187.i
	br label %bb2037.i

cond_false1063.i85:		; preds = %cond_false277.i
	ret void

bb2037.i:		; preds = %cond_true1032.i, %entry
	br bool false, label %bb205.i, label %cond_next2042.i

cond_next2042.i:		; preds = %bb2037.i
	ret void
}
