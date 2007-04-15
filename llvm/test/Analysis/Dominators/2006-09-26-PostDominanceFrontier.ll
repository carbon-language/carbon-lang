; RUN: llvm-upgrade < %s | llvm-as | opt -analyze -postdomfrontier \
; RUN:   -disable-verify
; END.
;
; ModuleID = '2006-09-26-PostDominanceFrontier.bc'
target endian = little
target pointersize = 64
target triple = "alphaev67-unknown-linux-gnu"
	%struct.FILE = type { int, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct._IO_marker*, %struct.FILE*, int, int, long, ushort, sbyte, [1 x sbyte], sbyte*, long, sbyte*, sbyte*, int, [44 x sbyte] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct.FILE*, int }
%TOP = external global ulong*		; <ulong**> [#uses=1]
%BOT = external global ulong*		; <ulong**> [#uses=1]
%str = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]

implementation   ; Functions:

declare void %fopen()

void %main(sbyte** %argv) {
entry:
	%netSelect.i507 = alloca ulong, align 8		; <ulong*> [#uses=0]
	%topStart.i = alloca ulong, align 8		; <ulong*> [#uses=0]
	%topEnd.i = alloca ulong, align 8		; <ulong*> [#uses=0]
	%botStart.i = alloca ulong, align 8		; <ulong*> [#uses=0]
	%botEnd.i = alloca ulong, align 8		; <ulong*> [#uses=0]
	%c1.i154 = alloca uint, align 4		; <uint*> [#uses=0]
	%b1.i155 = alloca uint, align 4		; <uint*> [#uses=0]
	%t1.i156 = alloca uint, align 4		; <uint*> [#uses=0]
	%c1.i = alloca uint, align 4		; <uint*> [#uses=0]
	%b1.i = alloca uint, align 4		; <uint*> [#uses=0]
	%t1.i = alloca uint, align 4		; <uint*> [#uses=0]
	%netSelect.i5 = alloca ulong, align 8		; <ulong*> [#uses=0]
	%netSelect.i = alloca ulong, align 8		; <ulong*> [#uses=0]
	%tmp2.i = getelementptr sbyte** %argv, int 1		; <sbyte**> [#uses=1]
	%tmp3.i4 = load sbyte** %tmp2.i		; <sbyte*> [#uses=0]
	call void %fopen( )
	br bool false, label %DimensionChannel.exit, label %bb.backedge.i

bb.backedge.i:		; preds = %entry
	ret void

DimensionChannel.exit:		; preds = %entry
	%tmp13.i137 = malloc ulong, uint 0		; <ulong*> [#uses=1]
	%tmp610.i = malloc ulong, uint 0		; <ulong*> [#uses=1]
	br label %cond_true.i143

cond_true.i143:		; preds = %cond_true.i143, %DimensionChannel.exit
	%tmp9.i140 = getelementptr ulong* %tmp13.i137, ulong 0		; <ulong*> [#uses=0]
	%tmp12.i = getelementptr ulong* %tmp610.i, ulong 0		; <ulong*> [#uses=0]
	br bool false, label %bb18.i144, label %cond_true.i143

bb18.i144:		; preds = %cond_true.i143
	call void %fopen( )
	%tmp76.i105 = malloc ulong, uint 0		; <ulong*> [#uses=3]
	%tmp674.i = malloc ulong, uint 0		; <ulong*> [#uses=2]
	%tmp1072.i = malloc ulong, uint 0		; <ulong*> [#uses=2]
	%tmp1470.i = malloc ulong, uint 0		; <ulong*> [#uses=1]
	br label %cond_true.i114

cond_true.i114:		; preds = %cond_true.i114, %bb18.i144
	%tmp17.i108 = getelementptr ulong* %tmp76.i105, ulong 0		; <ulong*> [#uses=0]
	%tmp20.i = getelementptr ulong* %tmp674.i, ulong 0		; <ulong*> [#uses=0]
	%tmp23.i111 = getelementptr ulong* %tmp1470.i, ulong 0		; <ulong*> [#uses=0]
	br bool false, label %cond_true40.i, label %cond_true.i114

cond_true40.i:		; preds = %cond_true40.i, %cond_true.i114
	%tmp33.i115 = getelementptr ulong* %tmp1072.i, ulong 0		; <ulong*> [#uses=0]
	br bool false, label %bb142.i, label %cond_true40.i

cond_next54.i:		; preds = %cond_true76.i
	%tmp57.i = getelementptr ulong* %tmp55.i, ulong 0		; <ulong*> [#uses=0]
	br bool false, label %bb64.i, label %bb69.i

bb64.i:		; preds = %cond_true76.i, %cond_next54.i
	%tmp67.i117 = getelementptr ulong* %tmp76.i105, ulong 0		; <ulong*> [#uses=0]
	br bool false, label %bb114.i, label %cond_true111.i

bb69.i:		; preds = %cond_next54.i
	br bool false, label %bb79.i, label %cond_true76.i

cond_true76.i:		; preds = %bb142.i, %bb69.i
	%tmp48.i = getelementptr ulong* %tmp46.i, ulong 0		; <ulong*> [#uses=0]
	br bool false, label %bb64.i, label %cond_next54.i

bb79.i:		; preds = %bb69.i
	br bool false, label %bb114.i, label %cond_true111.i

cond_true111.i:		; preds = %bb79.i, %bb64.i
	%tmp84.i127 = getelementptr ulong* %tmp46.i, ulong 0		; <ulong*> [#uses=0]
	ret void

bb114.i:		; preds = %bb142.i, %bb79.i, %bb64.i
	%tmp117.i = getelementptr ulong* %tmp76.i105, ulong 0		; <ulong*> [#uses=0]
	%tmp132.i131 = getelementptr ulong* %tmp674.i, ulong 0		; <ulong*> [#uses=0]
	%tmp122.i = getelementptr ulong* %tmp1072.i, ulong 0		; <ulong*> [#uses=0]
	ret void

bb142.i:		; preds = %cond_true40.i
	%tmp46.i = load ulong** %BOT		; <ulong*> [#uses=2]
	%tmp55.i = load ulong** %TOP		; <ulong*> [#uses=1]
	br bool false, label %bb114.i, label %cond_true76.i
}
