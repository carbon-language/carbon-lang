; RUN: opt < %s -indvars -S | FileCheck %s
; ModuleID = '<stdin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n32:64"
target triple = "x86_64-apple-darwin9.6"
@a = external global i32*		; <i32**> [#uses=3]
@b = external global i32*		; <i32**> [#uses=3]
@c = external global i32*		; <i32**> [#uses=3]
@d = external global i32*		; <i32**> [#uses=3]
@e = external global i32*		; <i32**> [#uses=3]
@f = external global i32*		; <i32**> [#uses=3]

define void @foo() nounwind {
; CHECK-LABEL: @foo(
; CHECK-NOT: sext
bb1.thread:
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %84, %bb1 ]		; <i32> [#uses=19]
	%0 = load i32*, i32** @a, align 8		; <i32*> [#uses=1]
	%1 = load i32*, i32** @b, align 8		; <i32*> [#uses=1]
	%2 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%3 = getelementptr i32, i32* %1, i64 %2		; <i32*> [#uses=1]
	%4 = load i32, i32* %3, align 1		; <i32> [#uses=1]
	%5 = load i32*, i32** @c, align 8		; <i32*> [#uses=1]
	%6 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%7 = getelementptr i32, i32* %5, i64 %6		; <i32*> [#uses=1]
	%8 = load i32, i32* %7, align 1		; <i32> [#uses=1]
	%9 = add i32 %8, %4		; <i32> [#uses=1]
	%10 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%11 = getelementptr i32, i32* %0, i64 %10		; <i32*> [#uses=1]
	store i32 %9, i32* %11, align 1
	%12 = load i32*, i32** @a, align 8		; <i32*> [#uses=1]
	%13 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%14 = load i32*, i32** @b, align 8		; <i32*> [#uses=1]
	%15 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%16 = sext i32 %15 to i64		; <i64> [#uses=1]
	%17 = getelementptr i32, i32* %14, i64 %16		; <i32*> [#uses=1]
	%18 = load i32, i32* %17, align 1		; <i32> [#uses=1]
	%19 = load i32*, i32** @c, align 8		; <i32*> [#uses=1]
	%20 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%21 = sext i32 %20 to i64		; <i64> [#uses=1]
	%22 = getelementptr i32, i32* %19, i64 %21		; <i32*> [#uses=1]
	%23 = load i32, i32* %22, align 1		; <i32> [#uses=1]
	%24 = add i32 %23, %18		; <i32> [#uses=1]
	%25 = sext i32 %13 to i64		; <i64> [#uses=1]
	%26 = getelementptr i32, i32* %12, i64 %25		; <i32*> [#uses=1]
	store i32 %24, i32* %26, align 1
	%27 = load i32*, i32** @a, align 8		; <i32*> [#uses=1]
	%28 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%29 = load i32*, i32** @b, align 8		; <i32*> [#uses=1]
	%30 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%31 = sext i32 %30 to i64		; <i64> [#uses=1]
	%32 = getelementptr i32, i32* %29, i64 %31		; <i32*> [#uses=1]
	%33 = load i32, i32* %32, align 1		; <i32> [#uses=1]
	%34 = load i32*, i32** @c, align 8		; <i32*> [#uses=1]
	%35 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%36 = sext i32 %35 to i64		; <i64> [#uses=1]
	%37 = getelementptr i32, i32* %34, i64 %36		; <i32*> [#uses=1]
	%38 = load i32, i32* %37, align 1		; <i32> [#uses=1]
	%39 = add i32 %38, %33		; <i32> [#uses=1]
	%40 = sext i32 %28 to i64		; <i64> [#uses=1]
	%41 = getelementptr i32, i32* %27, i64 %40		; <i32*> [#uses=1]
	store i32 %39, i32* %41, align 1
	%42 = load i32*, i32** @d, align 8		; <i32*> [#uses=1]
	%43 = load i32*, i32** @e, align 8		; <i32*> [#uses=1]
	%44 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%45 = getelementptr i32, i32* %43, i64 %44		; <i32*> [#uses=1]
	%46 = load i32, i32* %45, align 1		; <i32> [#uses=1]
	%47 = load i32*, i32** @f, align 8		; <i32*> [#uses=1]
	%48 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%49 = getelementptr i32, i32* %47, i64 %48		; <i32*> [#uses=1]
	%50 = load i32, i32* %49, align 1		; <i32> [#uses=1]
	%51 = add i32 %50, %46		; <i32> [#uses=1]
	%52 = sext i32 %i.0.reg2mem.0 to i64		; <i64> [#uses=1]
	%53 = getelementptr i32, i32* %42, i64 %52		; <i32*> [#uses=1]
	store i32 %51, i32* %53, align 1
	%54 = load i32*, i32** @d, align 8		; <i32*> [#uses=1]
	%55 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%56 = load i32*, i32** @e, align 8		; <i32*> [#uses=1]
	%57 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%58 = sext i32 %57 to i64		; <i64> [#uses=1]
	%59 = getelementptr i32, i32* %56, i64 %58		; <i32*> [#uses=1]
	%60 = load i32, i32* %59, align 1		; <i32> [#uses=1]
	%61 = load i32*, i32** @f, align 8		; <i32*> [#uses=1]
	%62 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=1]
	%63 = sext i32 %62 to i64		; <i64> [#uses=1]
	%64 = getelementptr i32, i32* %61, i64 %63		; <i32*> [#uses=1]
	%65 = load i32, i32* %64, align 1		; <i32> [#uses=1]
	%66 = add i32 %65, %60		; <i32> [#uses=1]
	%67 = sext i32 %55 to i64		; <i64> [#uses=1]
	%68 = getelementptr i32, i32* %54, i64 %67		; <i32*> [#uses=1]
	store i32 %66, i32* %68, align 1
	%69 = load i32*, i32** @d, align 8		; <i32*> [#uses=1]
	%70 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%71 = load i32*, i32** @e, align 8		; <i32*> [#uses=1]
	%72 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%73 = sext i32 %72 to i64		; <i64> [#uses=1]
	%74 = getelementptr i32, i32* %71, i64 %73		; <i32*> [#uses=1]
	%75 = load i32, i32* %74, align 1		; <i32> [#uses=1]
	%76 = load i32*, i32** @f, align 8		; <i32*> [#uses=1]
	%77 = add i32 %i.0.reg2mem.0, 2		; <i32> [#uses=1]
	%78 = sext i32 %77 to i64		; <i64> [#uses=1]
	%79 = getelementptr i32, i32* %76, i64 %78		; <i32*> [#uses=1]
	%80 = load i32, i32* %79, align 1		; <i32> [#uses=1]
	%81 = add i32 %80, %75		; <i32> [#uses=1]
	%82 = sext i32 %70 to i64		; <i64> [#uses=1]
	%83 = getelementptr i32, i32* %69, i64 %82		; <i32*> [#uses=1]
	store i32 %81, i32* %83, align 1
	%84 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%85 = icmp sgt i32 %84, 23646		; <i1> [#uses=1]
	br i1 %85, label %return, label %bb1

return:		; preds = %bb1
	ret void
}
