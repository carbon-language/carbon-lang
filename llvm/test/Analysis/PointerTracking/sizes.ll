; RUN: opt < %s -pointertracking -analyze | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@.str = internal constant [5 x i8] c"1234\00"		; <[5 x i8]*> [#uses=1]
@test1p = global i8* getelementptr ([5 x i8]* @.str, i32 0, i32 0), align 8		; <i8**> [#uses=1]
@test1a = global [5 x i8] c"1234\00", align 1		; <[5 x i8]*> [#uses=1]
@test2a = global [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5], align 4		; <[5 x i32]*> [#uses=2]
@test2p = global i32* getelementptr ([5 x i32]* @test2a, i32 0, i32 0), align 8		; <i32**> [#uses=1]
@test0p = common global i32* null, align 8		; <i32**> [#uses=1]
@test0i = common global i32 0, align 4		; <i32*> [#uses=1]

define i32 @foo0() nounwind {
entry:
	%tmp = load i32** @test0p		; <i32*> [#uses=1]
	%conv = bitcast i32* %tmp to i8*		; <i8*> [#uses=1]
	%call = tail call i32 @bar(i8* %conv) nounwind		; <i32> [#uses=1]
	%tmp1 = load i8** @test1p		; <i8*> [#uses=1]
	%call2 = tail call i32 @bar(i8* %tmp1) nounwind		; <i32> [#uses=1]
	%call3 = tail call i32 @bar(i8* getelementptr ([5 x i8]* @test1a, i32 0, i32 0)) nounwind		; <i32> [#uses=1]
	%call5 = tail call i32 @bar(i8* bitcast ([5 x i32]* @test2a to i8*)) nounwind		; <i32> [#uses=1]
	%tmp7 = load i32** @test2p		; <i32*> [#uses=1]
	%conv8 = bitcast i32* %tmp7 to i8*		; <i8*> [#uses=1]
	%call9 = tail call i32 @bar(i8* %conv8) nounwind		; <i32> [#uses=1]
	%call11 = tail call i32 @bar(i8* bitcast (i32* @test0i to i8*)) nounwind		; <i32> [#uses=1]
	%add = add i32 %call2, %call		; <i32> [#uses=1]
	%add4 = add i32 %add, %call3		; <i32> [#uses=1]
	%add6 = add i32 %add4, %call5		; <i32> [#uses=1]
	%add10 = add i32 %add6, %call9		; <i32> [#uses=1]
	%add12 = add i32 %add10, %call11		; <i32> [#uses=1]
	ret i32 %add12
}

declare i32 @bar(i8*)

define i32 @foo1(i32 %n) nounwind {
entry:
; CHECK: 'foo1':
	%test4a = alloca [10 x i8], align 1		; <[10 x i8]*> [#uses=1]
; CHECK: %test4a =
; CHECK: ==> 1 elements, 10 bytes allocated
	%test6a = alloca [10 x i32], align 4		; <[10 x i32]*> [#uses=1]
; CHECK: %test6a =
; CHECK: ==> 1 elements, 40 bytes allocated
	%vla = alloca i8, i32 %n, align 1		; <i8*> [#uses=1]
; CHECK: %vla =
; CHECK: ==> %n elements, %n bytes allocated
	%0 = shl i32 %n, 2		; <i32> [#uses=1]
	%vla7 = alloca i8, i32 %0, align 1		; <i8*> [#uses=1]
; CHECK: %vla7 =
; CHECK: ==> (4 * %n) elements, (4 * %n) bytes allocated
	%call = call i32 @bar(i8* %vla) nounwind		; <i32> [#uses=1]
	%arraydecay = getelementptr [10 x i8]* %test4a, i64 0, i64 0		; <i8*> [#uses=1]
	%call10 = call i32 @bar(i8* %arraydecay) nounwind		; <i32> [#uses=1]
	%call11 = call i32 @bar(i8* %vla7) nounwind		; <i32> [#uses=1]
	%ptrconv14 = bitcast [10 x i32]* %test6a to i8*		; <i8*> [#uses=1]
	%call15 = call i32 @bar(i8* %ptrconv14) nounwind		; <i32> [#uses=1]
	%add = add i32 %call10, %call		; <i32> [#uses=1]
	%add12 = add i32 %add, %call11		; <i32> [#uses=1]
	%add16 = add i32 %add12, %call15		; <i32> [#uses=1]
	ret i32 %add16
}

define i32 @foo2(i64 %n) nounwind {
entry:
	%call = tail call i8* @malloc(i64 %n)  ; <i8*> [#uses=1]
; CHECK: %call =
; CHECK: ==> %n elements, %n bytes allocated
	%call2 = tail call i8* @calloc(i64 2, i64 4) nounwind		; <i8*> [#uses=1]
; CHECK: %call2 =
; CHECK: ==> 8 elements, 8 bytes allocated
	%call4 = tail call i8* @realloc(i8* null, i64 16) nounwind		; <i8*> [#uses=1]
; CHECK: %call4 =
; CHECK: ==> 16 elements, 16 bytes allocated
	%call6 = tail call i32 @bar(i8* %call) nounwind		; <i32> [#uses=1]
	%call8 = tail call i32 @bar(i8* %call2) nounwind		; <i32> [#uses=1]
	%call10 = tail call i32 @bar(i8* %call4) nounwind		; <i32> [#uses=1]
	%add = add i32 %call8, %call6                   ; <i32> [#uses=1]
	%add11 = add i32 %add, %call10                ; <i32> [#uses=1]
	ret i32 %add11
}

declare noalias i8* @malloc(i64) nounwind

declare noalias i8* @calloc(i64, i64) nounwind

declare noalias i8* @realloc(i8* nocapture, i64) nounwind
