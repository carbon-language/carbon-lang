; RUN: opt < %s -scalarrepl -disable-output -stats |& grep "Number of aggregates converted to scalar"
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
	type { }		; type %0
	type { i8*, i32, i32, i16, i16, %2, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %2, %3*, i32, [3 x i8], [1 x i8], %2, i32, i64 }		; type %1
	type { i8*, i32 }		; type %2
	type opaque		; type %3
	type { i32 }		; type %4
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, %0*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0*, %0*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0* }
	%llvm.dbg.subprogram.type = type { i32, %0*, %0*, i8*, i8*, i8*, %0*, i32, %0*, i1, i1 }
	%llvm.dbg.subrange.type = type { i32, i64, i64 }
	%llvm.dbg.variable.type = type { i32, %0*, i8*, %0*, i32, %0* }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
internal constant [8 x i8] c"PR491.c\00", section "llvm.metadata"		; <[8 x i8]*>:0 [#uses=1]
internal constant [77 x i8] c"/Volumes/Nanpura/mainline/llvm/projects/llvm-test/SingleSource/Regression/C/\00", section "llvm.metadata"		; <[77 x i8]*>:1 [#uses=1]
internal constant [55 x i8] c"4.2.1 (Based on Apple Inc. build 5641) (LLVM build 00)\00", section "llvm.metadata"		; <[55 x i8]*>:2 [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 1, i8* getelementptr ([8 x i8]* @0, i32 0, i32 0), i8* getelementptr ([77 x i8]* @1, i32 0, i32 0), i8* getelementptr ([55 x i8]* @2, i32 0, i32 0), i1 true, i1 false, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*>:3 [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @3, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
internal constant [5 x i8] c"char\00", section "llvm.metadata"		; <[5 x i8]*>:4 [#uses=1]
@llvm.dbg.basictype5 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @4, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 8, i64 8, i64 0, i32 0, i32 6 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458790, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 8, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype5 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype6 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
internal constant [13 x i8] c"unsigned int\00", section "llvm.metadata"		; <[13 x i8]*>:5 [#uses=1]
@llvm.dbg.basictype8 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @5, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 7 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array = internal constant [3 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype6 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype8 to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
internal constant [12 x i8] c"assert_fail\00", section "llvm.metadata"		; <[12 x i8]*>:6 [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([12 x i8]* @6, i32 0, i32 0), i8* getelementptr ([12 x i8]* @6, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 4, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to %0*), i1 true, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=0]
internal constant [2 x i8] c"l\00", section "llvm.metadata"		; <[2 x i8]*>:7 [#uses=1]
@__stderrp = external global %1*		; <%1**> [#uses=4]
internal constant [35 x i8] c"assertion failed in line %u: '%s'\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[35 x i8]*>:8 [#uses=1]
@llvm.dbg.array13 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite14 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array13 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
internal constant [5 x i8] c"test\00", section "llvm.metadata"		; <[5 x i8]*>:9 [#uses=1]
@llvm.dbg.subprogram16 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @9, i32 0, i32 0), i8* getelementptr ([5 x i8]* @9, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 10, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite14 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
internal constant [9 x i8] c"long int\00", section "llvm.metadata"		; <[9 x i8]*>:10 [#uses=1]
@llvm.dbg.basictype21 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @10, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.derivedtype22 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram16 to %0*), i8* getelementptr ([2 x i8]* @7, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 20, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype21 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.subrange = internal constant %llvm.dbg.subrange.type { i32 458785, i64 0, i64 3 }, section "llvm.metadata"		; <%llvm.dbg.subrange.type*> [#uses=1]
@llvm.dbg.array23 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.subrange.type* @llvm.dbg.subrange to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
internal constant [14 x i8] c"unsigned char\00", section "llvm.metadata"		; <[14 x i8]*>:11 [#uses=1]
@llvm.dbg.basictype25 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @11, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 8, i64 8, i64 0, i32 0, i32 8 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.composite26 = internal constant %llvm.dbg.composite.type { i32 458753, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype25 to %0*), %0* bitcast ([1 x %0*]* @llvm.dbg.array23 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
internal constant [2 x i8] c"c\00", section "llvm.metadata"		; <[2 x i8]*>:12 [#uses=1]
@llvm.dbg.derivedtype28 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram16 to %0*), i8* getelementptr ([2 x i8]* @12, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 20, i64 32, i64 8, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite26 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array29 = internal constant [2 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype22 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype28 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite30 = internal constant %llvm.dbg.composite.type { i32 458775, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram16 to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 20, i64 32, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array29 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
internal constant [2 x i8] c"u\00", section "llvm.metadata"		; <[2 x i8]*>:13 [#uses=1]
@llvm.dbg.variable32 = internal constant %llvm.dbg.variable.type { i32 459008, %0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram16 to %0*), i8* getelementptr ([2 x i8]* @13, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 20, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite30 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=1]
internal constant [11 x i8] c"u.l == 128\00", section "__TEXT,__cstring,cstring_literals"		; <[11 x i8]*>:14 [#uses=1]
internal constant [8 x i8] c"u.l < 0\00", section "__TEXT,__cstring,cstring_literals"		; <[8 x i8]*>:15 [#uses=1]
@llvm.dbg.array35 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite36 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array35 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*>:16 [#uses=1]
@llvm.dbg.subprogram38 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @16, i32 0, i32 0), i8* getelementptr ([5 x i8]* @16, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 28, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite36 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]

declare void @llvm.dbg.func.start(%0*) nounwind readnone

declare void @llvm.dbg.declare(%0*, %0*) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, %0*) nounwind readnone

declare i32 @fprintf(%1* nocapture, i8* nocapture, ...) nounwind

declare void @llvm.dbg.region.end(%0*) nounwind readnone

define i32 @test(i32) nounwind {
; <label>:1
	%2 = alloca %4, align 8		; <%4*> [#uses=7]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram16 to %0*))
	%3 = bitcast %4* %2 to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %3, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable32 to %0*))
	call void @llvm.dbg.stoppoint(i32 21, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%4 = getelementptr %4* %2, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %4, align 8
	%5 = bitcast %4* %2 to i8*		; <i8*> [#uses=1]
	store i8 -128, i8* %5, align 8
	call void @llvm.dbg.stoppoint(i32 22, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%6 = getelementptr %4* %2, i32 0, i32 0		; <i32*> [#uses=1]
	%7 = load i32* %6, align 8		; <i32> [#uses=1]
	%8 = icmp eq i32 %7, 128		; <i1> [#uses=1]
	br i1 %8, label %12, label %9

; <label>:9		; preds = %1
	call void @llvm.dbg.stoppoint(i32 5, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	%10 = load %1** @__stderrp, align 4		; <%1*> [#uses=1]
	%11 = call i32 (%1*, i8*, ...)* @fprintf(%1* %10, i8* getelementptr ([35 x i8]* @8, i32 0, i32 0), i32 22, i8* getelementptr ([11 x i8]* @14, i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 6, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	br label %12

; <label>:12		; preds = %9, %1
	%.0 = phi i32 [ 0, %9 ], [ 1, %1 ]		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 22, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%13 = and i32 %.0, %0		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 23, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%14 = getelementptr %4* %2, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %14, align 8
	%15 = bitcast %4* %2 to [4 x i8]*		; <[4 x i8]*> [#uses=1]
	%16 = getelementptr [4 x i8]* %15, i32 0, i32 3		; <i8*> [#uses=1]
	store i8 -128, i8* %16, align 1
	call void @llvm.dbg.stoppoint(i32 24, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%17 = getelementptr %4* %2, i32 0, i32 0		; <i32*> [#uses=1]
	%18 = load i32* %17, align 8		; <i32> [#uses=1]
	%19 = icmp slt i32 %18, 0		; <i1> [#uses=1]
	br i1 %19, label %23, label %20

; <label>:20		; preds = %12
	call void @llvm.dbg.stoppoint(i32 5, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	%21 = load %1** @__stderrp, align 4		; <%1*> [#uses=1]
	%22 = call i32 (%1*, i8*, ...)* @fprintf(%1* %21, i8* getelementptr ([35 x i8]* @8, i32 0, i32 0), i32 24, i8* getelementptr ([8 x i8]* @15, i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 6, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	br label %23

; <label>:23		; preds = %20, %12
	%.01 = phi i32 [ 0, %20 ], [ 1, %12 ]		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 24, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%24 = and i32 %.01, %13		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 25, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram16 to %0*))
	ret i32 %24
}

define i32 @main() nounwind {
; <label>:0
	%1 = alloca %4, align 8		; <%4*> [#uses=7]
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram38 to %0*))
	call void @llvm.dbg.stoppoint(i32 29, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%2 = bitcast %4* %1 to %0*		; <%0*> [#uses=1]
	call void @llvm.dbg.declare(%0* %2, %0* bitcast (%llvm.dbg.variable.type* @llvm.dbg.variable32 to %0*)) nounwind
	call void @llvm.dbg.stoppoint(i32 21, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	%3 = getelementptr %4* %1, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %3, align 8
	%4 = bitcast %4* %1 to i8*		; <i8*> [#uses=1]
	store i8 -128, i8* %4, align 8
	call void @llvm.dbg.stoppoint(i32 22, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	%5 = getelementptr %4* %1, i32 0, i32 0		; <i32*> [#uses=1]
	%6 = load i32* %5, align 8		; <i32> [#uses=1]
	%7 = icmp eq i32 %6, 128		; <i1> [#uses=1]
	br i1 %7, label %11, label %8

; <label>:8		; preds = %0
	call void @llvm.dbg.stoppoint(i32 5, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	%9 = load %1** @__stderrp, align 4		; <%1*> [#uses=1]
	%10 = call i32 (%1*, i8*, ...)* @fprintf(%1* %9, i8* getelementptr ([35 x i8]* @8, i32 0, i32 0), i32 22, i8* getelementptr ([11 x i8]* @14, i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 6, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	br label %11

; <label>:11		; preds = %8, %0
	%.0.i = phi i32 [ 0, %8 ], [ 1, %0 ]		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 23, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	%12 = getelementptr %4* %1, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %12, align 8
	%13 = bitcast %4* %1 to [4 x i8]*		; <[4 x i8]*> [#uses=1]
	%14 = getelementptr [4 x i8]* %13, i32 0, i32 3		; <i8*> [#uses=1]
	store i8 -128, i8* %14, align 1
	call void @llvm.dbg.stoppoint(i32 24, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	%15 = getelementptr %4* %1, i32 0, i32 0		; <i32*> [#uses=1]
	%16 = load i32* %15, align 8		; <i32> [#uses=1]
	%17 = icmp slt i32 %16, 0		; <i1> [#uses=1]
	br i1 %17, label %test.exit, label %18

; <label>:18		; preds = %11
	call void @llvm.dbg.stoppoint(i32 5, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	%19 = load %1** @__stderrp, align 4		; <%1*> [#uses=1]
	%20 = call i32 (%1*, i8*, ...)* @fprintf(%1* %19, i8* getelementptr ([35 x i8]* @8, i32 0, i32 0), i32 24, i8* getelementptr ([8 x i8]* @15, i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	call void @llvm.dbg.stoppoint(i32 6, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	br label %test.exit

test.exit:		; preds = %18, %11
	%.01.i = phi i32 [ 0, %18 ], [ 1, %11 ]		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 24, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	%21 = and i32 %.01.i, %.0.i		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 25, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*)) nounwind
	%tmp = xor i32 %21, 1		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 29, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram38 to %0*))
	ret i32 %tmp
}
