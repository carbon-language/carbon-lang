; RUN: llc < %s -mtriple=i386-unknown-linux-gnu | grep cmpxchgl | not grep eax
; PR4076

	type { i8, i8, i8 }		; type %0
	type { i32, i8** }		; type %1
	type { %3* }		; type %2
	type { %4 }		; type %3
	type { %5 }		; type %4
	type { %6, i32, %7 }		; type %5
	type { i8* }		; type %6
	type { i32, [12 x i8] }		; type %7
	type { %9 }		; type %8
	type { %10, %11*, i8 }		; type %9
	type { %11* }		; type %10
	type { i32, %6, i8*, %12, %13*, i8, i32, %28, %29, i32, %30, i32, i32, i32, i8*, i8*, i8, i8 }		; type %11
	type { %13* }		; type %12
	type { %14, i32, %13*, %21 }		; type %13
	type { %15, %16 }		; type %14
	type { i32 (...)** }		; type %15
	type { %17, i8* (i32)*, void (i8*)*, i8 }		; type %16
	type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %18 }		; type %17
	type { %19* }		; type %18
	type { i32, %20**, i32, %20**, i8** }		; type %19
	type { i32 (...)**, i32 }		; type %20
	type { %22, %25*, i8, i8, %17*, %26*, %27*, %27* }		; type %21
	type { i32 (...)**, i32, i32, i32, i32, i32, %23*, %24, [8 x %24], i32, %24*, %18 }		; type %22
	type { %23*, void (i32, %22*, i32)*, i32, i32 }		; type %23
	type { i8*, i32 }		; type %24
	type { i32 (...)**, %21 }		; type %25
	type { %20, i32*, i8, i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8 }		; type %26
	type { %20 }		; type %27
	type { void (%9*)*, i32 }		; type %28
	type { %15* }		; type %29
	type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8* }		; type %30
@AtomicOps_Internalx86CPUFeatures = external global %0		; <%0*> [#uses=1]
internal constant [19 x i8] c"xxxxxxxxxxxxxxxxxx\00"		; <[19 x i8]*>:0 [#uses=1]
internal constant [47 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00"		; <[47 x i8]*>:1 [#uses=1]

define i8** @func6(i8 zeroext, i32, i32, %1*) nounwind {
; <label>:4
	%5 = alloca i32, align 4		; <i32*> [#uses=2]
	%6 = alloca i32, align 4		; <i32*> [#uses=2]
	%7 = alloca %2, align 8		; <%2*> [#uses=3]
	%8 = alloca %8, align 8		; <%8*> [#uses=2]
	br label %17

; <label>:9		; preds = %17
	%10 = getelementptr %1* %3, i32 %19, i32 0		; <i32*> [#uses=1]
	%11 = load i32* %10, align 4		; <i32> [#uses=1]
	%12 = icmp eq i32 %11, %2		; <i1> [#uses=1]
	br i1 %12, label %13, label %16

; <label>:13		; preds = %9
	%14 = getelementptr %1* %3, i32 %19, i32 1		; <i8***> [#uses=1]
	%15 = load i8*** %14, align 4		; <i8**> [#uses=1]
	ret i8** %15

; <label>:16		; preds = %9
	%indvar.next13 = add i32 %18, 1		; <i32> [#uses=1]
	br label %17

; <label>:17		; preds = %16, %4
	%18 = phi i32 [ 0, %4 ], [ %indvar.next13, %16 ]		; <i32> [#uses=2]
	%19 = add i32 %18, %1		; <i32> [#uses=3]
	%20 = icmp sgt i32 %19, 3		; <i1> [#uses=1]
	br i1 %20, label %21, label %9

; <label>:21		; preds = %17
	call void @func5()
	%22 = getelementptr %1* %3, i32 0, i32 0		; <i32*> [#uses=1]
	%23 = load i32* %22, align 4		; <i32> [#uses=1]
	%24 = icmp eq i32 %23, 0		; <i1> [#uses=1]
	br i1 %24, label %._crit_edge, label %._crit_edge1

._crit_edge1:		; preds = %._crit_edge1, %21
	%25 = phi i32 [ 0, %21 ], [ %26, %._crit_edge1 ]		; <i32> [#uses=1]
	%26 = add i32 %25, 1		; <i32> [#uses=4]
	%27 = getelementptr %1* %3, i32 %26, i32 0		; <i32*> [#uses=1]
	%28 = load i32* %27, align 4		; <i32> [#uses=1]
	%29 = icmp ne i32 %28, 0		; <i1> [#uses=1]
	%30 = icmp ne i32 %26, 4		; <i1> [#uses=1]
	%31 = and i1 %29, %30		; <i1> [#uses=1]
	br i1 %31, label %._crit_edge1, label %._crit_edge

._crit_edge:		; preds = %._crit_edge1, %21
	%32 = phi i32 [ 0, %21 ], [ %26, %._crit_edge1 ]		; <i32> [#uses=3]
	%33 = call i8* @pthread_getspecific(i32 0) nounwind		; <i8*> [#uses=2]
	%34 = icmp ne i8* %33, null		; <i1> [#uses=1]
	%35 = icmp eq i8 %0, 0		; <i1> [#uses=1]
	%36 = or i1 %34, %35		; <i1> [#uses=1]
	br i1 %36, label %._crit_edge4, label %37

; <label>:37		; preds = %._crit_edge
	%38 = call i8* @func2(i32 2048)		; <i8*> [#uses=4]
	call void @llvm.memset.i32(i8* %38, i8 0, i32 2048, i32 4)
	%39 = call i32 @pthread_setspecific(i32 0, i8* %38) nounwind		; <i32> [#uses=2]
	store i32 %39, i32* %5
	store i32 0, i32* %6
	%40 = icmp eq i32 %39, 0		; <i1> [#uses=1]
	br i1 %40, label %41, label %43

; <label>:41		; preds = %37
	%42 = getelementptr %2* %7, i32 0, i32 0		; <%3**> [#uses=1]
	store %3* null, %3** %42, align 8
	br label %._crit_edge4

; <label>:43		; preds = %37
	%44 = call %3* @func1(i32* %5, i32* %6, i8* getelementptr ([47 x i8]* @1, i32 0, i32 0))		; <%3*> [#uses=2]
	%45 = getelementptr %2* %7, i32 0, i32 0		; <%3**> [#uses=1]
	store %3* %44, %3** %45, align 8
	%46 = icmp eq %3* %44, null		; <i1> [#uses=1]
	br i1 %46, label %._crit_edge4, label %47

; <label>:47		; preds = %43
	call void @func4(%8* %8, i8* getelementptr ([19 x i8]* @0, i32 0, i32 0), i32 165, %2* %7)
	call void @func3(%8* %8) noreturn
	unreachable

._crit_edge4:		; preds = %43, %41, %._crit_edge
	%48 = phi i8* [ %38, %41 ], [ %33, %._crit_edge ], [ %38, %43 ]		; <i8*> [#uses=2]
	%49 = bitcast i8* %48 to i8**		; <i8**> [#uses=3]
	%50 = icmp ne i8* %48, null		; <i1> [#uses=1]
	%51 = icmp slt i32 %32, 4		; <i1> [#uses=1]
	%52 = and i1 %50, %51		; <i1> [#uses=1]
	br i1 %52, label %53, label %._crit_edge6

; <label>:53		; preds = %._crit_edge4
	%54 = getelementptr %1* %3, i32 %32, i32 0		; <i32*> [#uses=1]
	%55 = call i32 asm sideeffect "lock; cmpxchgl $1,$2", "={ax},q,*m,0,~{dirflag},~{fpsr},~{flags},~{memory}"(i32 %2, i32* %54, i32 0) nounwind		; <i32> [#uses=1]
	%56 = load i8* getelementptr (%0* @AtomicOps_Internalx86CPUFeatures, i32 0, i32 0), align 8		; <i8> [#uses=1]
	%57 = icmp eq i8 %56, 0		; <i1> [#uses=1]
	br i1 %57, label %._crit_edge7, label %58

; <label>:58		; preds = %53
	call void asm sideeffect "lfence", "~{dirflag},~{fpsr},~{flags},~{memory}"() nounwind
	br label %._crit_edge7

._crit_edge7:		; preds = %58, %53
	%59 = icmp eq i32 %55, 0		; <i1> [#uses=1]
	br i1 %59, label %60, label %._crit_edge6

._crit_edge6:		; preds = %._crit_edge7, %._crit_edge4
	ret i8** %49

; <label>:60		; preds = %._crit_edge7
	%61 = getelementptr %1* %3, i32 %32, i32 1		; <i8***> [#uses=1]
	store i8** %49, i8*** %61, align 4
	ret i8** %49
}

declare %3* @func1(i32* nocapture, i32* nocapture, i8*)

declare void @func5()

declare void @func4(%8*, i8*, i32, %2*)

declare void @func3(%8*) noreturn

declare i8* @pthread_getspecific(i32) nounwind

declare i8* @func2(i32)

declare void @llvm.memset.i32(i8* nocapture, i8, i32, i32) nounwind

declare i32 @pthread_setspecific(i32, i8*) nounwind
