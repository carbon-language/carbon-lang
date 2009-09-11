; RUN: opt < %s -instcombine -S | grep store | count 5
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
	type { }		; type %0
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, %0*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0*, %0*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0* }
	%llvm.dbg.subprogram.type = type { i32, %0*, %0*, i8*, i8*, i8*, %0*, i32, %0*, i1, i1 }
	%struct.Matrix = type { float*, i32, i32, i32, i32 }
@llvm.dbg.compile_units = internal constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [15 x i8] c"himenobmtxpa.c\00", section "llvm.metadata"		; <[15 x i8]*> [#uses=1]
@.str1 = internal constant [74 x i8] c"/Volumes/MacOS9/gcc/llvm/projects/llvm-test/SingleSource/Benchmarks/Misc/\00", section "llvm.metadata"		; <[74 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5641) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 1, i8* getelementptr ([15 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([74 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [6 x i8] c"float\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str3, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 4 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str5 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype6 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str5, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.subprograms = internal constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str90 = internal constant [4 x i8] c"Mat\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.derivedtype92 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str93 = internal constant [2 x i8] c"m\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype94 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([2 x i8]* @.str93, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 46, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype92 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str95 = internal constant [6 x i8] c"mnums\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype96 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str95, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 47, i64 32, i64 32, i64 32, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str97 = internal constant [6 x i8] c"mrows\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype98 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str97, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 48, i64 32, i64 32, i64 64, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str99 = internal constant [6 x i8] c"mcols\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype100 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str99, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 49, i64 32, i64 32, i64 96, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str101 = internal constant [6 x i8] c"mdeps\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@llvm.dbg.derivedtype102 = internal constant %llvm.dbg.derivedtype.type { i32 458765, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([6 x i8]* @.str101, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 50, i64 32, i64 32, i64 128, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array103 = internal constant [5 x %0*] [%0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype94 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype96 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype98 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype100 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype102 to %0*)], section "llvm.metadata"		; <[5 x %0*]*> [#uses=1]
@llvm.dbg.composite104 = internal constant %llvm.dbg.composite.type { i32 458771, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str90, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 45, i64 160, i64 32, i64 0, i32 0, %0* null, %0* bitcast ([5 x %0*]* @llvm.dbg.array103 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str105 = internal constant [7 x i8] c"Matrix\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype106 = internal constant %llvm.dbg.derivedtype.type { i32 458774, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str105, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 54, i64 0, i64 0, i64 0, i32 0, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite104 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.derivedtype107 = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype106 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array108 = internal constant [6 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*), %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype107 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*)], section "llvm.metadata"		; <[6 x %0*]*> [#uses=1]
@llvm.dbg.composite109 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([6 x %0*]* @llvm.dbg.array108 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str110 = internal constant [7 x i8] c"newMat\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.subprogram111 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str110, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.str110, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 195, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite109 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (%struct.Matrix*, i32, i32, i32, i32)* @newMat to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i32 @newMat(%struct.Matrix* %Mat, i32 %mnums, i32 %mrows, i32 %mcols, i32 %mdeps) nounwind {
entry:
	call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram111 to %0*))
	call void @llvm.dbg.stoppoint(i32 196, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = getelementptr %struct.Matrix* %Mat, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %mnums, i32* %0, align 4
	call void @llvm.dbg.stoppoint(i32 197, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%1 = getelementptr %struct.Matrix* %Mat, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 %mrows, i32* %1, align 4
	call void @llvm.dbg.stoppoint(i32 198, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%2 = getelementptr %struct.Matrix* %Mat, i32 0, i32 3		; <i32*> [#uses=1]
	store i32 %mcols, i32* %2, align 4
	call void @llvm.dbg.stoppoint(i32 199, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%3 = getelementptr %struct.Matrix* %Mat, i32 0, i32 4		; <i32*> [#uses=1]
	store i32 %mdeps, i32* %3, align 4
	call void @llvm.dbg.stoppoint(i32 201, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%4 = mul i32 %mnums, %mrows		; <i32> [#uses=1]
	%5 = mul i32 %4, %mcols		; <i32> [#uses=1]
	%6 = mul i32 %5, %mdeps		; <i32> [#uses=1]
	%7 = malloc float, i32 %6		; <float*> [#uses=2]
	%8 = getelementptr %struct.Matrix* %Mat, i32 0, i32 0		; <float**> [#uses=1]
	store float* %7, float** %8, align 4
	call void @llvm.dbg.stoppoint(i32 204, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%9 = icmp ne float* %7, null		; <i1> [#uses=1]
	%10 = zext i1 %9 to i32		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 204, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram111 to %0*))
	ret i32 %10
}

declare void @llvm.dbg.func.start(%0*) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, %0*) nounwind readnone

declare void @llvm.dbg.region.end(%0*) nounwind readnone
