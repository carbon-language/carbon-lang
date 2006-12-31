; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify | llvm-dis | grep br | grep return.i.bb8_crit_edge | grep false

%str = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=1]

implementation   ; Functions:

declare int %sprintf(sbyte*, sbyte*, ...)

int %main() {
entry:
	br label %cond_true.outer

cond_true.outer:		; preds = %cond_true.i, %entry
	%i.0.0.ph = phi int [ 0, %entry ], [ %tmp5, %cond_true.i ]		; <int> [#uses=1]
	%j.0.0.ph = phi int [ 0, %entry ], [ %tmp312, %cond_true.i ]		; <int> [#uses=2]
	br label %cond_true

cond_true:		; preds = %return.i, %cond_true.outer
	%indvar.ui = phi uint [ 0, %cond_true.outer ], [ %indvar.next, %return.i ]		; <uint> [#uses=2]
	%indvar = cast uint %indvar.ui to int		; <int> [#uses=1]
	%i.0.0 = add int %indvar, %i.0.0.ph		; <int> [#uses=3]
	%savedstack = call sbyte* %llvm.stacksave( )		; <sbyte*> [#uses=2]
	%tmp.i = seteq int %i.0.0, 0		; <bool> [#uses=1]
	%tmp5 = add int %i.0.0, 1		; <int> [#uses=3]
	br bool %tmp.i, label %return.i, label %cond_true.i

cond_true.i:		; preds = %cond_true
	%tmp.i = alloca [1000 x sbyte]		; <[1000 x sbyte]*> [#uses=1]
	%tmp.sub.i = getelementptr [1000 x sbyte]* %tmp.i, int 0, int 0		; <sbyte*> [#uses=2]
	%tmp4.i = call int (sbyte*, sbyte*, ...)* %sprintf( sbyte* %tmp.sub.i, sbyte* getelementptr ([4 x sbyte]* %str, int 0, uint 0), int %i.0.0 )		; <int> [#uses=0]
	%tmp.i = load sbyte* %tmp.sub.i		; <sbyte> [#uses=1]
	%tmp7.i = cast sbyte %tmp.i to int		; <int> [#uses=1]
	call void %llvm.stackrestore( sbyte* %savedstack )
	%tmp312 = add int %tmp7.i, %j.0.0.ph		; <int> [#uses=2]
	%tmp19 = setgt int %tmp5, 9999		; <bool> [#uses=1]
	br bool %tmp19, label %bb8, label %cond_true.outer

return.i:		; preds = %cond_true
	call void %llvm.stackrestore( sbyte* %savedstack )
	%tmp21 = setgt int %tmp5, 9999		; <bool> [#uses=1]
	%indvar.next = add uint %indvar.ui, 1		; <uint> [#uses=1]
	br bool %tmp21, label %bb8, label %cond_true

bb8:		; preds = %return.i, %cond_true.i
	%j.0.1 = phi int [ %j.0.0.ph, %return.i ], [ %tmp312, %cond_true.i ]		; <int> [#uses=1]
	%tmp10 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([4 x sbyte]* %str, int 0, uint 0), int %j.0.1 )		; <int> [#uses=0]
	ret int undef
}

declare int %printf(sbyte*, ...)

declare sbyte* %llvm.stacksave()

declare void %llvm.stackrestore(sbyte*)
