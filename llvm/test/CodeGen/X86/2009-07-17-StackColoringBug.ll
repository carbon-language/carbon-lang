; RUN: llc < %s -mtriple=i386-pc-linux-gnu -disable-fp-elim -color-ss-with-regs | not grep dil
; PR4552

target triple = "i386-pc-linux-gnu"
@g_8 = internal global i32 0		; <i32*> [#uses=1]
@g_72 = internal global i32 0		; <i32*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (i32, i8, i8)* @uint84 to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i32 @uint84(i32 %p_15, i8 signext %p_17, i8 signext %p_19) nounwind {
entry:
	%g_72.promoted = load i32* @g_72		; <i32> [#uses=1]
	%g_8.promoted = load i32* @g_8		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %func_40.exit, %entry
	%g_8.tmp.1 = phi i32 [ %g_8.promoted, %entry ], [ %g_8.tmp.0, %func_40.exit ]		; <i32> [#uses=3]
	%g_72.tmp.1 = phi i32 [ %g_72.promoted, %entry ], [ %g_72.tmp.0, %func_40.exit ]		; <i32> [#uses=3]
	%retval12.i4.i.i = trunc i32 %g_8.tmp.1 to i8		; <i8> [#uses=2]
	%0 = trunc i32 %g_72.tmp.1 to i8		; <i8> [#uses=2]
	%1 = mul i8 %retval12.i4.i.i, %0		; <i8> [#uses=1]
	%2 = icmp eq i8 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %bb2.i.i, label %bb.i.i

bb.i.i:		; preds = %bb
	%3 = sext i8 %0 to i32		; <i32> [#uses=1]
	%4 = and i32 %3, 50295		; <i32> [#uses=1]
	%5 = icmp eq i32 %4, 0		; <i1> [#uses=1]
	br i1 %5, label %bb2.i.i, label %func_55.exit.i

bb2.i.i:		; preds = %bb.i.i, %bb
	br label %func_55.exit.i

func_55.exit.i:		; preds = %bb2.i.i, %bb.i.i
	%g_72.tmp.2 = phi i32 [ 1, %bb2.i.i ], [ %g_72.tmp.1, %bb.i.i ]		; <i32> [#uses=1]
	%6 = phi i32 [ 1, %bb2.i.i ], [ %g_72.tmp.1, %bb.i.i ]		; <i32> [#uses=1]
	%7 = trunc i32 %6 to i8		; <i8> [#uses=2]
	%8 = mul i8 %7, %retval12.i4.i.i		; <i8> [#uses=1]
	%9 = icmp eq i8 %8, 0		; <i1> [#uses=1]
	br i1 %9, label %bb2.i4.i, label %bb.i3.i

bb.i3.i:		; preds = %func_55.exit.i
	%10 = sext i8 %7 to i32		; <i32> [#uses=1]
	%11 = and i32 %10, 50295		; <i32> [#uses=1]
	%12 = icmp eq i32 %11, 0		; <i1> [#uses=1]
	br i1 %12, label %bb2.i4.i, label %func_40.exit

bb2.i4.i:		; preds = %bb.i3.i, %func_55.exit.i
	br label %func_40.exit

func_40.exit:		; preds = %bb2.i4.i, %bb.i3.i
	%g_72.tmp.0 = phi i32 [ 1, %bb2.i4.i ], [ %g_72.tmp.2, %bb.i3.i ]		; <i32> [#uses=1]
	%phitmp = icmp sgt i32 %g_8.tmp.1, 0		; <i1> [#uses=1]
	%g_8.tmp.0 = select i1 %phitmp, i32 %g_8.tmp.1, i32 1		; <i32> [#uses=1]
	br label %bb
}
