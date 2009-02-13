; RUN: llvm-as < %s | opt -gvn -debug |& grep {REMOVING NONLOCAL LOAD} | count 1

; The bug this is testing for is that BasicAA considered:
;   alias(%a = noalias call, %b = gep(%a, %unknown))
; to be NoAlias instead of MayAlias.

@.str = external constant [3 x i8]		; <[3 x i8]*> [#uses=1]
@.str1 = external constant [23 x i8]		; <[23 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind {
entry:
	%0 = tail call i8* @calloc(i32 11, i32 4) nounwind		; <i8*> [#uses=1]
	%1 = bitcast i8* %0 to i32*		; <i32*> [#uses=7]
	%2 = tail call i8* @calloc(i32 11, i32 4) nounwind		; <i8*> [#uses=2]
	%3 = bitcast i8* %2 to i32*		; <i32*> [#uses=9]
	%4 = tail call i8* @calloc(i32 11, i32 4) nounwind		; <i8*> [#uses=1]
	%5 = bitcast i8* %4 to i32*		; <i32*> [#uses=3]
	br label %bb3.i

bb3.i:		; preds = %bb3.i, %entry
	%i.0.reg2mem.0.i = phi i32 [ 0, %entry ], [ %7, %bb3.i ]		; <i32> [#uses=3]
	%6 = getelementptr i32* %3, i32 %i.0.reg2mem.0.i		; <i32*> [#uses=1]
	store i32 %i.0.reg2mem.0.i, i32* %6, align 4
	%7 = add i32 %i.0.reg2mem.0.i, 1		; <i32> [#uses=2]
	%8 = icmp slt i32 %7, 11		; <i1> [#uses=1]
	br i1 %8, label %bb3.i, label %bb8.i.loopexit

bb5.i:		; preds = %bb29.i
	%9 = icmp sgt i32 %didpr.1.reg2mem.0.ph.i.ph, 29		; <i1> [#uses=1]
	br i1 %9, label %bb11.i.outer, label %bb8.loopexit21.i

bb8.loopexit21.i:		; preds = %bb5.i
	%phitmp = add i32 %didpr.1.reg2mem.0.ph.i.ph, 1		; <i32> [#uses=1]
	br label %bb8.i.outer

bb8.i.loopexit:		; preds = %bb3.i
	br label %bb8.i.outer

bb8.i.outer:		; preds = %bb8.i.loopexit, %bb8.loopexit21.i
	%didpr.1.reg2mem.0.ph.i.ph = phi i32 [ %phitmp, %bb8.loopexit21.i ], [ 1, %bb8.i.loopexit ]		; <i32> [#uses=2]
	%r.1.reg2mem.0.ph.i.ph = phi i32 [ %r.2.i, %bb8.loopexit21.i ], [ 11, %bb8.i.loopexit ]		; <i32> [#uses=1]
	%flipsMax.1.reg2mem.0.ph.i.ph = phi i32 [ %flipsMax.0.ph.i.ph, %bb8.loopexit21.i ], [ 0, %bb8.i.loopexit ]		; <i32> [#uses=1]
	br label %bb8.i

bb8.i:		; preds = %bb8.i, %bb8.i.outer
	%i.1.reg2mem.0.i = phi i32 [ %14, %bb8.i ], [ 0, %bb8.i.outer ]		; <i32> [#uses=2]
	%10 = getelementptr i32* %3, i32 %i.1.reg2mem.0.i		; <i32*> [#uses=1]
	%11 = load i32* %10, align 4		; <i32> [#uses=1]
	%12 = add i32 %11, 1		; <i32> [#uses=1]
	%13 = tail call i32 (i8*, ...)* @printf(i8* noalias getelementptr ([3 x i8]* @.str, i32 0, i32 0), i32 %12) nounwind		; <i32> [#uses=0]
	%14 = add i32 %i.1.reg2mem.0.i, 1		; <i32> [#uses=2]
	%15 = icmp slt i32 %14, 11		; <i1> [#uses=1]
	br i1 %15, label %bb8.i, label %bb9.i

bb9.i:		; preds = %bb8.i
	%16 = tail call i32 @putchar(i32 10) nounwind		; <i32> [#uses=0]
	br label %bb11.i.outer

bb11.i.outer:		; preds = %bb9.i, %bb5.i
	%flipsMax.1.reg2mem.1.ph.i.ph = phi i32 [ %flipsMax.1.reg2mem.0.ph.i.ph, %bb9.i ], [ %flipsMax.0.ph.i.ph, %bb5.i ]		; <i32> [#uses=4]
	%r.0.i.ph = phi i32 [ %r.1.reg2mem.0.ph.i.ph, %bb9.i ], [ %r.2.i, %bb5.i ]		; <i32> [#uses=1]
	br label %bb11.i

bb10.i:		; preds = %bb11.i
	%17 = add i32 %r.0.i, -1		; <i32> [#uses=1]
	%18 = getelementptr i32* %5, i32 %17		; <i32*> [#uses=1]
	store i32 %r.0.i, i32* %18, align 4
	%19 = add i32 %r.0.i, -1		; <i32> [#uses=1]
	br label %bb11.i

bb11.i:		; preds = %bb10.i, %bb11.i.outer
	%r.0.i = phi i32 [ %19, %bb10.i ], [ %r.0.i.ph, %bb11.i.outer ]		; <i32> [#uses=5]
	%20 = icmp eq i32 %r.0.i, 1		; <i1> [#uses=1]
	br i1 %20, label %bb12.i, label %bb10.i

bb12.i:		; preds = %bb11.i
	%21 = load i32* %3, align 4		; <i32> [#uses=1]
	%22 = icmp eq i32 %21, 0		; <i1> [#uses=1]
	br i1 %22, label %bb24.i.preheader, label %bb13.i

bb13.i:		; preds = %bb12.i
	%23 = getelementptr i8* %2, i32 40		; <i8*> [#uses=1]
	%24 = bitcast i8* %23 to i32*		; <i32*> [#uses=1]
	%25 = load i32* %24, align 4		; <i32> [#uses=1]
	%26 = icmp eq i32 %25, 10		; <i1> [#uses=1]
	br i1 %26, label %bb24.i.preheader, label %bb16.i.preheader

bb16.i.preheader:		; preds = %bb13.i
	br label %bb16.i

bb16.i:		; preds = %bb16.i, %bb16.i.preheader
	%i.2.reg2mem.0.i = phi i32 [ %30, %bb16.i ], [ 1, %bb16.i.preheader ]		; <i32> [#uses=3]
	%27 = getelementptr i32* %3, i32 %i.2.reg2mem.0.i		; <i32*> [#uses=1]
	%28 = load i32* %27, align 4		; <i32> [#uses=1]
	%29 = getelementptr i32* %1, i32 %i.2.reg2mem.0.i		; <i32*> [#uses=1]
	store i32 %28, i32* %29, align 4
	%30 = add i32 %i.2.reg2mem.0.i, 1		; <i32> [#uses=2]
	%31 = icmp slt i32 %30, 11		; <i1> [#uses=1]
	br i1 %31, label %bb16.i, label %bb17.i

bb17.i:		; preds = %bb16.i
	%32 = load i32* %3, align 4		; <i32> [#uses=2]
	br label %bb20.i.outer

bb20.i.outer:		; preds = %bb21.i, %bb17.i
	%k.0.ph.i.ph = phi i32 [ %32, %bb17.i ], [ %43, %bb21.i ]		; <i32> [#uses=3]
	%flips.0.ph.i.ph = phi i32 [ 0, %bb17.i ], [ %41, %bb21.i ]		; <i32> [#uses=1]
	%j.0.in.i.ph = phi i32 [ %32, %bb17.i ], [ %43, %bb21.i ]		; <i32> [#uses=1]
	br label %bb20.i

bb19.i:		; preds = %bb20.i
	%33 = getelementptr i32* %1, i32 %i.3.i		; <i32*> [#uses=1]
	%34 = load i32* %33, align 4		; <i32> [#uses=1]
	%35 = getelementptr i32* %1, i32 %j.0.i		; <i32*> [#uses=1]
	%36 = load i32* %35, align 4		; <i32> [#uses=1]
	%37 = getelementptr i32* %1, i32 %i.3.i		; <i32*> [#uses=1]
	store i32 %36, i32* %37, align 4
	%38 = getelementptr i32* %1, i32 %j.0.i		; <i32*> [#uses=1]
	store i32 %34, i32* %38, align 4
	%39 = add i32 %i.3.i, 1		; <i32> [#uses=1]
	br label %bb20.i

bb20.i:		; preds = %bb19.i, %bb20.i.outer
	%i.3.i = phi i32 [ %39, %bb19.i ], [ 1, %bb20.i.outer ]		; <i32> [#uses=4]
	%j.0.in.i = phi i32 [ %j.0.i, %bb19.i ], [ %j.0.in.i.ph, %bb20.i.outer ]		; <i32> [#uses=1]
	%j.0.i = add i32 %j.0.in.i, -1		; <i32> [#uses=4]
	%40 = icmp slt i32 %i.3.i, %j.0.i		; <i1> [#uses=1]
	br i1 %40, label %bb19.i, label %bb21.i

bb21.i:		; preds = %bb20.i
	%41 = add i32 %flips.0.ph.i.ph, 1		; <i32> [#uses=3]
	%42 = getelementptr i32* %1, i32 %k.0.ph.i.ph		; <i32*> [#uses=1]
	%43 = load i32* %42, align 4		; <i32> [#uses=3]
	%44 = getelementptr i32* %1, i32 %k.0.ph.i.ph		; <i32*> [#uses=1]
	store i32 %k.0.ph.i.ph, i32* %44, align 4
	%45 = icmp eq i32 %43, 0		; <i1> [#uses=1]
	br i1 %45, label %bb22.i, label %bb20.i.outer

bb22.i:		; preds = %bb21.i
	%46 = icmp slt i32 %flipsMax.1.reg2mem.1.ph.i.ph, %41		; <i1> [#uses=1]
	br i1 %46, label %bb23.i, label %bb24.i.preheader

bb23.i:		; preds = %bb22.i
	br label %bb24.i.preheader

bb24.i.preheader:		; preds = %bb23.i, %bb22.i, %bb13.i, %bb12.i
	%flipsMax.0.ph.i.ph = phi i32 [ %flipsMax.1.reg2mem.1.ph.i.ph, %bb22.i ], [ %flipsMax.1.reg2mem.1.ph.i.ph, %bb13.i ], [ %flipsMax.1.reg2mem.1.ph.i.ph, %bb12.i ], [ %41, %bb23.i ]		; <i32> [#uses=3]
	br label %bb24.i

bb24.i:		; preds = %bb30.i, %bb24.i.preheader
	%r.2.i = phi i32 [ %60, %bb30.i ], [ %r.0.i, %bb24.i.preheader ]		; <i32> [#uses=8]
	%47 = icmp eq i32 %r.2.i, 11		; <i1> [#uses=1]
	br i1 %47, label %fannkuch.exit, label %bb26.i

bb26.i:		; preds = %bb24.i
	%48 = load i32* %3, align 4		; <i32> [#uses=1]
	br label %bb28.i

bb27.i:		; preds = %bb28.i
	%49 = add i32 %i.4.i, 1		; <i32> [#uses=2]
	%50 = getelementptr i32* %3, i32 %49		; <i32*> [#uses=1]
	%51 = load i32* %50, align 4		; <i32> [#uses=1]
	%52 = getelementptr i32* %3, i32 %i.4.i		; <i32*> [#uses=1]
	store i32 %51, i32* %52, align 4
	br label %bb28.i

bb28.i:		; preds = %bb27.i, %bb26.i
	%i.4.i = phi i32 [ 0, %bb26.i ], [ %49, %bb27.i ]		; <i32> [#uses=3]
	%53 = icmp slt i32 %i.4.i, %r.2.i		; <i1> [#uses=1]
	br i1 %53, label %bb27.i, label %bb29.i

bb29.i:		; preds = %bb28.i
	%54 = getelementptr i32* %3, i32 %r.2.i		; <i32*> [#uses=1]
	store i32 %48, i32* %54, align 4
	%55 = getelementptr i32* %5, i32 %r.2.i		; <i32*> [#uses=1]
	%56 = load i32* %55, align 4		; <i32> [#uses=1]
	%57 = add i32 %56, -1		; <i32> [#uses=2]
	%58 = getelementptr i32* %5, i32 %r.2.i		; <i32*> [#uses=1]
	store i32 %57, i32* %58, align 4
	%59 = icmp sgt i32 %57, 0		; <i1> [#uses=1]
	br i1 %59, label %bb5.i, label %bb30.i

bb30.i:		; preds = %bb29.i
	%60 = add i32 %r.2.i, 1		; <i32> [#uses=1]
	br label %bb24.i

fannkuch.exit:		; preds = %bb24.i
	%61 = tail call i32 (i8*, ...)* @printf(i8* noalias getelementptr ([23 x i8]* @.str1, i32 0, i32 0), i32 11, i32 %flipsMax.0.ph.i.ph) nounwind		; <i32> [#uses=0]
	ret i32 0
}

declare noalias i8* @calloc(i32, i32) nounwind

declare i32 @printf(i8* nocapture, ...) nounwind

declare i32 @putchar(i32) nounwind
