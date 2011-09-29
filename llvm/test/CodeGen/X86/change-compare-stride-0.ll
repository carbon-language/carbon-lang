; RUN: llc < %s -march=x86 -enable-lsr-nested | FileCheck %s
;
; Nested LSR is required to optimize this case.
; We do not expect to see this form of IR without -enable-iv-rewrite.

define void @borf(i8* nocapture %in, i8* nocapture %out) nounwind {
; CHECK: borf:
; CHECK-NOT: inc
; CHECK-NOT: leal 1(
; CHECK-NOT: leal -1(
; CHECK: decl
; CHECK-NEXT: cmpl $-478
; CHECK: ret

bb4.thread:
	br label %bb2.outer

bb2.outer:		; preds = %bb4, %bb4.thread
	%indvar18 = phi i32 [ 0, %bb4.thread ], [ %indvar.next28, %bb4 ]		; <i32> [#uses=3]
	%tmp34 = mul i32 %indvar18, 65535		; <i32> [#uses=1]
	%i.0.reg2mem.0.ph = add i32 %tmp34, 639		; <i32> [#uses=1]
	%0 = and i32 %i.0.reg2mem.0.ph, 65535		; <i32> [#uses=1]
	%1 = mul i32 %0, 480		; <i32> [#uses=1]
	%tmp20 = mul i32 %indvar18, -478		; <i32> [#uses=1]
	br label %bb2

bb2:		; preds = %bb2, %bb2.outer
	%indvar = phi i32 [ 0, %bb2.outer ], [ %indvar.next, %bb2 ]		; <i32> [#uses=3]
	%ctg2 = getelementptr i8* %out, i32 %tmp20		; <i8*> [#uses=1]
	%tmp21 = ptrtoint i8* %ctg2 to i32		; <i32> [#uses=1]
	%tmp23 = sub i32 %tmp21, %indvar		; <i32> [#uses=1]
	%out_addr.0.reg2mem.0 = inttoptr i32 %tmp23 to i8*		; <i8*> [#uses=1]
	%tmp25 = mul i32 %indvar, 65535		; <i32> [#uses=1]
	%j.0.reg2mem.0 = add i32 %tmp25, 479		; <i32> [#uses=1]
	%2 = and i32 %j.0.reg2mem.0, 65535		; <i32> [#uses=1]
	%3 = add i32 %1, %2		; <i32> [#uses=9]
	%4 = add i32 %3, -481		; <i32> [#uses=1]
	%5 = getelementptr i8* %in, i32 %4		; <i8*> [#uses=1]
	%6 = load i8* %5, align 1		; <i8> [#uses=1]
	%7 = add i32 %3, -480		; <i32> [#uses=1]
	%8 = getelementptr i8* %in, i32 %7		; <i8*> [#uses=1]
	%9 = load i8* %8, align 1		; <i8> [#uses=1]
	%10 = add i32 %3, -479		; <i32> [#uses=1]
	%11 = getelementptr i8* %in, i32 %10		; <i8*> [#uses=1]
	%12 = load i8* %11, align 1		; <i8> [#uses=1]
	%13 = add i32 %3, -1		; <i32> [#uses=1]
	%14 = getelementptr i8* %in, i32 %13		; <i8*> [#uses=1]
	%15 = load i8* %14, align 1		; <i8> [#uses=1]
	%16 = getelementptr i8* %in, i32 %3		; <i8*> [#uses=1]
	%17 = load i8* %16, align 1		; <i8> [#uses=1]
	%18 = add i32 %3, 1		; <i32> [#uses=1]
	%19 = getelementptr i8* %in, i32 %18		; <i8*> [#uses=1]
	%20 = load i8* %19, align 1		; <i8> [#uses=1]
	%21 = add i32 %3, 481		; <i32> [#uses=1]
	%22 = getelementptr i8* %in, i32 %21		; <i8*> [#uses=1]
	%23 = load i8* %22, align 1		; <i8> [#uses=1]
	%24 = add i32 %3, 480		; <i32> [#uses=1]
	%25 = getelementptr i8* %in, i32 %24		; <i8*> [#uses=1]
	%26 = load i8* %25, align 1		; <i8> [#uses=1]
	%27 = add i32 %3, 479		; <i32> [#uses=1]
	%28 = getelementptr i8* %in, i32 %27		; <i8*> [#uses=1]
	%29 = load i8* %28, align 1		; <i8> [#uses=1]
	%30 = add i8 %9, %6		; <i8> [#uses=1]
	%31 = add i8 %30, %12		; <i8> [#uses=1]
	%32 = add i8 %31, %15		; <i8> [#uses=1]
	%33 = add i8 %32, %17		; <i8> [#uses=1]
	%34 = add i8 %33, %20		; <i8> [#uses=1]
	%35 = add i8 %34, %23		; <i8> [#uses=1]
	%36 = add i8 %35, %26		; <i8> [#uses=1]
	%37 = add i8 %36, %29		; <i8> [#uses=1]
	store i8 %37, i8* %out_addr.0.reg2mem.0, align 1
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 478		; <i1> [#uses=1]
	br i1 %exitcond, label %bb4, label %bb2

bb4:		; preds = %bb2
	%indvar.next28 = add i32 %indvar18, 1		; <i32> [#uses=2]
	%exitcond29 = icmp eq i32 %indvar.next28, 638		; <i1> [#uses=1]
	br i1 %exitcond29, label %return, label %bb2.outer

return:		; preds = %bb4
	ret void
}
