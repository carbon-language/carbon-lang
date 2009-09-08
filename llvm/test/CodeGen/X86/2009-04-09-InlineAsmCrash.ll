; RUN: llc < %s 
; rdar://6774324
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"
	type <{ i32, %1 }>		; type %0
	type <{ [216 x i8] }>		; type %1
	type <{ %3, %4*, %28*, i64, i32, %6, %6, i32, i32, i32, i32, void (i8*, i32)*, i8*, %29*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i8*], i32, %30, i32, %24, %4*, %4*, i64, i64, i32, i32, void (i32, %2*)*, i32, i32, i32, i32, i32, i32, i32, i32, %24, i64, i64, i64, i64, i64, %21, i32, i32, %21, i32, %31*, %3, %33, %34, %9*, i32, i32, %3, %3, %35, %41*, %42*, %11, i32, i32, i32, i8, i8, i8, i8, %69*, %69, %9*, %9*, [11 x %61], %3, i8*, i32, i64, i64, i32, i32, i32, i64 }>		; type %2
	type <{ %3*, %3* }>		; type %3
	type <{ %3, i32, %2*, %2*, %2*, %5*, i32, i32, %21, i64, i64, i64, i32, %22, %9*, %6, %4*, %23 }>		; type %4
	type <{ %3, %3, %4*, %4*, i32, %6, %9*, %9*, %5*, %20* }>		; type %5
	type <{ %7, i16, i8, i8, %8 }>		; type %6
	type <{ i32 }>		; type %7
	type <{ i8*, i8*, [2 x i32], i16, i8, i8, i8*, i8, i8, i8, i8, i8* }>		; type %8
	type <{ %10, %13, %15, i32, i32, i32, i32, %9*, %9*, %16*, i32, %17*, i64, i32 }>		; type %9
	type <{ i32, i32, %11 }>		; type %10
	type <{ %12 }>		; type %11
	type <{ [12 x i8] }>		; type %12
	type <{ %14 }>		; type %13
	type <{ [40 x i8] }>		; type %14
	type <{ [4 x i8] }>		; type %15
	type <{ %15, %15 }>		; type %16
	type <{ %17*, %17*, %9*, i32, %18*, %19* }>		; type %17
	type opaque		; type %18
	type <{ i32, i32, %9*, %9*, i32, i32 }>		; type %19
	type <{ %5*, %20*, %20*, %20* }>		; type %20
	type <{ %3, %3*, void (i8*, i8*)*, i8*, i8*, i64 }>		; type %21
	type <{ i32, [4 x i32], i32, i32, [128 x %3] }>		; type %22
	type <{ %24, %24, %24, %24*, %24*, %24*, %25, %26, %27, i32, i32, i8* }>		; type %23
	type <{ i64, i32, i32, i32 }>		; type %24
	type <{ i32, i32 }>		; type %25
	type <{ i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32 }>		; type %26
	type <{ [16 x %17*], i32 }>		; type %27
	type <{ i8, i8, i8, i8, %7, %3 }>		; type %28
	type <{ i32, %11*, i8*, i8*, %11* }>		; type %29
	type <{ i32, i32, i32, i32, i64 }>		; type %30
	type <{ %32*, %3, %3, i32, i32, i32, %5* }>		; type %31
	type opaque		; type %32
	type <{ [44 x i8] }>		; type %33
	type <{ %17* }>		; type %34
	type <{ %36, %36*, i32, [4 x %40], i32, i32, i64, i32 }>		; type %35
	type <{ i8*, %0*, %37*, i64, %39, i32, %39, %6, i64, i64, i8*, i32 }>		; type %36
	type <{ i32, i32, i8, i8, i8, i8, i8, i8, i8, i8, %38 }>		; type %37
	type <{ i16, i16, i8, i8, i16, i32, i16, i16, i32, i16, i16, i32, i32, [8 x [8 x i16]], [8 x [16 x i16]], [96 x i8] }>		; type %38
	type <{ i8, i8, i8, i8, i8, i8, i8, i8 }>		; type %39
	type <{ i64 }>		; type %40
	type <{ %11, i32, i32, i32, %42*, %3, i8*, %3, %5*, %32*, i32, i32, i32, i32, i32, i32, i32, %59, %60, i64, i64, i32, %11, %9*, %9*, %9*, [11 x %61], %9*, %9*, %9*, %9*, %9*, [3 x %9*], %62*, %3, %3, i32, i32, %9*, %9*, i32, %67*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, %68*, [2 x i32], i64, i64, i32 }>		; type %41
	type <{ %43, %44, %47*, i64, i64, i64, i32, %11, %54, %46*, %46*, i32, i32, i32, i32, i32, i32, i32 }>		; type %42
	type <{ i16, i8, i8, i32, i32 }>		; type %43
	type <{ %45, i32, i32 }>		; type %44
	type <{ %46*, %46*, i64, i64 }>		; type %45
	type <{ %45, %15, i64, i8, i8, i8, i8, i16, i16 }>		; type %46
	type <{ i64*, i64, %48*, i32, i32, i32, %6, %53, i32, i64, i64*, i64*, %48*, %48*, %48*, i32 }>		; type %47
	type <{ %3, %43, i64, %49*, i32, i32, i32, i32, %48*, %48*, i64, %50*, i64, %52*, i32, i16, i16, i8, i8, i8, i8, %3, %3, i64, i32, i32, i32, i8*, i32, i8, i8, i8, i8, %3 }>		; type %48
	type <{ %3, %3, %49*, %48*, i64, i8, i8, i8, i8, i32, i8, i8, i8, i8 }>		; type %49
	type <{ i32, %51* }>		; type %50
	type <{ void (%50*)*, void (%50*)*, i32 (%50*, %52*, i32)*, i32 (%50*)*, i32 (%50*, i64, i32, i32, i32*)*, i32 (%50*, i64, i32, i64*, i32*, i32, i32, i32)*, i32 (%50*, i64, i32)*, i32 (%50*, i64, i64, i32)*, i32 (%50*, i64, i64, i32)*, i32 (%50*, i32)*, i32 (%50*)*, i8* }>		; type %51
	type <{ i32, %48* }>		; type %52
	type <{ i32, i32, i32 }>		; type %53
	type <{ %11, %55*, i32, %53, i64 }>		; type %54
	type <{ %3, i32, i32, i32, i32, i32, [64 x i8], %56 }>		; type %55
	type <{ %57, %58, %58 }>		; type %56
	type <{ i64, i64, i64, i64, i64 }>		; type %57
	type <{ i64, i64, i64, i64, i64, i64, i64, i64 }>		; type %58
	type <{ [2 x i32] }>		; type %59
	type <{ [8 x i32] }>		; type %60
	type <{ %9*, i32, i32, i32 }>		; type %61
	type <{ %11, i32, %11, i32, i32, %63*, i32, %64*, %65, i32, i32, i32, i32, %41* }>		; type %62
	type <{ %10*, i32, %15, %15 }>		; type %63
	type opaque		; type %64
	type <{ i32, %66*, %66*, %66**, %66*, %66** }>		; type %65
	type <{ %63, i32, %62*, %66*, %66* }>		; type %66
	type <{ i32, i32, [0 x %39] }>		; type %67
	type opaque		; type %68
	type <{ %69*, void (%69*, %2*)* }>		; type %69
	type <{ %70*, %2*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i32, i32, i32, i32, i32, i32, i32, %71, i32, i32, i64, i64, i64, %72, i8*, i8*, %73, %4*, %79*, %81*, %39*, %84, i32, i32, i32, i8*, i32, i32, i32, i32, i32, i32, i32, i64*, i32, i64*, i8*, i32, [256 x i32], i64, i64, %86, %77*, i64, i64, %88*, %2*, %2* }>		; type %70
	type <{ %3, i64, i32, i32 }>		; type %71
	type <{ i64, i64, i64 }>		; type %72
	type <{ %73*, %73*, %73*, %73*, %74*, %75*, %76*, %70*, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, [3 x %78*], i8*, i8* }>		; type %73
	type <{ %74*, %74*, %75*, %76*, %73*, i32, i32, i32, i32, i32, i8*, i8* }>		; type %74
	type <{ %75*, %73*, %74*, %76*, i32, i32, i32, i32, %78*, i8*, i8* }>		; type %75
	type <{ %76*, %73*, %74*, %75*, i32, i32, i32, i32, i8*, i8*, %77* }>		; type %76
	type opaque		; type %77
	type <{ %78*, %75*, i8, i8, i8, i8, i16, i16, i16, i8, i8, i32, [0 x %73*] }>		; type %78
	type <{ i32, i32, i32, [20 x %80] }>		; type %79
	type <{ i64*, i8* }>		; type %80
	type <{ [256 x %39], [19 x %39], i8, i8, i8, i8, i8, i8, i8, i8, %82, i8, i8, i8, i8, i8, i8, i8, i8, %82, %83 }>		; type %81
	type <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, i16 }>		; type %82
	type <{ [16 x i64], i64 }>		; type %83
	type <{ %82*, %85, %85, %39*, i32 }>		; type %84
	type <{ i16, %39* }>		; type %85
	type <{ %87, i8* }>		; type %86
	type <{ i32, i32, i32, i8, i8, i16, i32, i32, i32, i32, i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }>		; type %87
	type <{ i64, i64, i32, i32, i32, i32 }>		; type %88
	type <{ i32, i32, i32, i32, i32, i32, i32 }>		; type %89
@kernel_stack_size = external global i32		; <i32*> [#uses=1]

define void @test(%0*) nounwind {
	%2 = tail call %2* asm sideeffect "mov %gs:${1:P},$0", "=r,i,~{dirflag},~{fpsr},~{flags}"(i32 ptrtoint (%2** getelementptr (%70* null, i32 0, i32 1) to i32)) nounwind		; <%2*> [#uses=1]
	%3 = getelementptr %2* %2, i32 0, i32 15		; <i32*> [#uses=1]
	%4 = load i32* %3		; <i32> [#uses=2]
	%5 = icmp eq i32 %4, 0		; <i1> [#uses=1]
	br i1 %5, label %47, label %6

; <label>:6		; preds = %1
	%7 = load i32* @kernel_stack_size		; <i32> [#uses=1]
	%8 = add i32 %7, %4		; <i32> [#uses=1]
	%9 = inttoptr i32 %8 to %89*		; <%89*> [#uses=12]
	%10 = tail call %2* asm sideeffect "mov %gs:${1:P},$0", "=r,i,~{dirflag},~{fpsr},~{flags}"(i32 ptrtoint (%2** getelementptr (%70* null, i32 0, i32 1) to i32)) nounwind		; <%2*> [#uses=1]
	%11 = getelementptr %2* %10, i32 0, i32 65, i32 1		; <%36**> [#uses=1]
	%12 = load %36** %11		; <%36*> [#uses=1]
	%13 = getelementptr %36* %12, i32 0, i32 1		; <%0**> [#uses=1]
	%14 = load %0** %13		; <%0*> [#uses=1]
	%15 = icmp eq %0* %14, %0		; <i1> [#uses=1]
	br i1 %15, label %40, label %16

; <label>:16		; preds = %6
	%17 = getelementptr %0* %0, i32 0, i32 1		; <%1*> [#uses=1]
	%18 = getelementptr %89* %9, i32 -1, i32 0		; <i32*> [#uses=1]
	%19 = getelementptr %0* %0, i32 0, i32 1, i32 0, i32 32		; <i8*> [#uses=1]
	%20 = bitcast i8* %19 to i32*		; <i32*> [#uses=1]
	%21 = load i32* %20		; <i32> [#uses=1]
	store i32 %21, i32* %18
	%22 = getelementptr %89* %9, i32 -1, i32 1		; <i32*> [#uses=1]
	%23 = ptrtoint %1* %17 to i32		; <i32> [#uses=1]
	store i32 %23, i32* %22
	%24 = getelementptr %89* %9, i32 -1, i32 2		; <i32*> [#uses=1]
	%25 = getelementptr %0* %0, i32 0, i32 1, i32 0, i32 24		; <i8*> [#uses=1]
	%26 = bitcast i8* %25 to i32*		; <i32*> [#uses=1]
	%27 = load i32* %26		; <i32> [#uses=1]
	store i32 %27, i32* %24
	%28 = getelementptr %89* %9, i32 -1, i32 3		; <i32*> [#uses=1]
	%29 = getelementptr %0* %0, i32 0, i32 1, i32 0, i32 16		; <i8*> [#uses=1]
	%30 = bitcast i8* %29 to i32*		; <i32*> [#uses=1]
	%31 = load i32* %30		; <i32> [#uses=1]
	store i32 %31, i32* %28
	%32 = getelementptr %89* %9, i32 -1, i32 4		; <i32*> [#uses=1]
	%33 = getelementptr %0* %0, i32 0, i32 1, i32 0, i32 20		; <i8*> [#uses=1]
	%34 = bitcast i8* %33 to i32*		; <i32*> [#uses=1]
	%35 = load i32* %34		; <i32> [#uses=1]
	store i32 %35, i32* %32
	%36 = getelementptr %89* %9, i32 -1, i32 5		; <i32*> [#uses=1]
	%37 = getelementptr %0* %0, i32 0, i32 1, i32 0, i32 56		; <i8*> [#uses=1]
	%38 = bitcast i8* %37 to i32*		; <i32*> [#uses=1]
	%39 = load i32* %38		; <i32> [#uses=1]
	store i32 %39, i32* %36
	ret void

; <label>:40		; preds = %6
	%41 = getelementptr %89* %9, i32 -1, i32 0		; <i32*> [#uses=1]
	tail call void asm sideeffect "movl %ebx, $0", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* %41) nounwind
	%42 = getelementptr %89* %9, i32 -1, i32 1		; <i32*> [#uses=1]
	tail call void asm sideeffect "movl %esp, $0", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* %42) nounwind
	%43 = getelementptr %89* %9, i32 -1, i32 2		; <i32*> [#uses=1]
	tail call void asm sideeffect "movl %ebp, $0", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* %43) nounwind
	%44 = getelementptr %89* %9, i32 -1, i32 3		; <i32*> [#uses=1]
	tail call void asm sideeffect "movl %edi, $0", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* %44) nounwind
	%45 = getelementptr %89* %9, i32 -1, i32 4		; <i32*> [#uses=1]
	tail call void asm sideeffect "movl %esi, $0", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* %45) nounwind
	%46 = getelementptr %89* %9, i32 -1, i32 5		; <i32*> [#uses=1]
	tail call void asm sideeffect "movl $$1f, $0\0A1:", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* %46) nounwind
	ret void

; <label>:47		; preds = %1
	ret void
}
