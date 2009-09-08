; RUN: llc < %s -march=x86 -mtriple=i386-apple-darwin8 -pre-alloc-split

define i32 @main() nounwind {
bb4.i.thread:
	br label %bb5.i4

bb16:		; preds = %bb111.i
	%phitmp = add i32 %indvar.reg2mem.4, 1		; <i32> [#uses=2]
	switch i32 %indvar.reg2mem.4, label %bb100.i [
		i32 0, label %bb5.i4
		i32 1, label %bb5.i4
		i32 2, label %bb5.i4
		i32 5, label %bb.i14.i
		i32 6, label %bb.i14.i
		i32 7, label %bb.i14.i
	]

bb5.i4:		; preds = %bb16, %bb16, %bb16, %bb4.i.thread
	br i1 false, label %bb102.i, label %bb103.i

bb.i14.i:		; preds = %bb16, %bb16, %bb16
	%0 = malloc [600 x i32]		; <[600 x i32]*> [#uses=0]
	%1 = icmp eq i32 %phitmp, 7		; <i1> [#uses=1]
	%tl.0.i = select i1 %1, float 1.000000e+02, float 1.000000e+00		; <float> [#uses=1]
	%2 = icmp eq i32 %phitmp, 8		; <i1> [#uses=1]
	%tu.0.i = select i1 %2, float 1.000000e+02, float 1.000000e+00		; <float> [#uses=1]
	br label %bb30.i

bb30.i:		; preds = %bb36.i, %bb.i14.i
	%i.1173.i = phi i32 [ 0, %bb.i14.i ], [ %indvar.next240.i, %bb36.i ]		; <i32> [#uses=3]
	%3 = icmp eq i32 0, %i.1173.i		; <i1> [#uses=1]
	br i1 %3, label %bb33.i, label %bb34.i

bb33.i:		; preds = %bb30.i
	store float %tl.0.i, float* null, align 4
	br label %bb36.i

bb34.i:		; preds = %bb30.i
	%4 = icmp eq i32 0, %i.1173.i		; <i1> [#uses=1]
	br i1 %4, label %bb35.i, label %bb36.i

bb35.i:		; preds = %bb34.i
	store float %tu.0.i, float* null, align 4
	br label %bb36.i

bb36.i:		; preds = %bb35.i, %bb34.i, %bb33.i
	%indvar.next240.i = add i32 %i.1173.i, 1		; <i32> [#uses=1]
	br label %bb30.i

bb100.i:		; preds = %bb16
	ret i32 0

bb102.i:		; preds = %bb5.i4
	br label %bb103.i

bb103.i:		; preds = %bb102.i, %bb5.i4
	%indvar.reg2mem.4 = phi i32 [ 0, %bb5.i4 ], [ 0, %bb102.i ]		; <i32> [#uses=2]
	%n.0.reg2mem.1.i = phi i32 [ 0, %bb102.i ], [ 0, %bb5.i4 ]		; <i32> [#uses=1]
	%5 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %5, label %bb111.i, label %bb108.i

bb108.i:		; preds = %bb103.i
	ret i32 0

bb111.i:		; preds = %bb103.i
	%6 = icmp sgt i32 %n.0.reg2mem.1.i, 7		; <i1> [#uses=1]
	br i1 %6, label %bb16, label %bb112.i

bb112.i:		; preds = %bb111.i
	unreachable
}
