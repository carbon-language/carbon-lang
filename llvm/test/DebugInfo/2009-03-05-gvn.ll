; RUN: opt < %s -gvn -S | grep {load } | count 1
; ModuleID = 'db2-before.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	type { }		; type %0
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, %0*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0*, %0*, i32 }
	%llvm.dbg.derivedtype.type = type { i32, %0*, i8*, %0*, i32, i64, i64, i64, i32, %0* }
	%llvm.dbg.subprogram.type = type { i32, %0*, %0*, i8*, i8*, i8*, %0*, i32, %0*, i1, i1 }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [12 x i8] c"mt19937ar.c\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@.str1 = internal constant [34 x i8] c"/developer/home2/zsth/test/debug/\00", section "llvm.metadata"		; <[34 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5641) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to %0*), i32 1, i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([34 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@mti = internal global i32 625		; <i32*> [#uses=14]
@.str3 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str5 = internal constant [18 x i8] c"long unsigned int\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.basictype6 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([18 x i8]* @.str5, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 7 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array = internal constant [2 x %0*] [%0* null, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*)], section "llvm.metadata"		; <[2 x %0*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([2 x %0*]* @llvm.dbg.array to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str7 = internal constant [13 x i8] c"init_genrand\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([13 x i8]* @.str7, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str7, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 58, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@mt = internal global [624 x i32] zeroinitializer, align 32		; <[624 x i32]*> [#uses=29]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458767, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array9 = internal constant [3 x %0*] [%0* null, %0* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to %0*), %0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*)], section "llvm.metadata"		; <[3 x %0*]*> [#uses=1]
@llvm.dbg.composite10 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([3 x %0*]* @llvm.dbg.array9 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str11 = internal constant [14 x i8] c"init_by_array\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram12 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str11, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str11, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 77, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite10 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array23 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite24 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array23 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str25 = internal constant [14 x i8] c"genrand_int32\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram26 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str25, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str25, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 103, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite24 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@mag01.1661 = internal constant [2 x i32] [i32 0, i32 -1727483681]		; <[2 x i32]*> [#uses=3]
@.str35 = internal constant [9 x i8] c"long int\00", section "llvm.metadata"		; <[9 x i8]*> [#uses=1]
@llvm.dbg.basictype36 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([9 x i8]* @.str35, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array37 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype36 to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite38 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array37 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str39 = internal constant [14 x i8] c"genrand_int31\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram40 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str39, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str39, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 141, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite38 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str41 = internal constant [7 x i8] c"double\00", section "llvm.metadata"		; <[7 x i8]*> [#uses=1]
@llvm.dbg.basictype42 = internal constant %llvm.dbg.basictype.type { i32 458788, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([7 x i8]* @.str41, i32 0, i32 0), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 64, i64 64, i64 0, i32 0, i32 4 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array43 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype42 to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite44 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array43 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str45 = internal constant [14 x i8] c"genrand_real1\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram46 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str45, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str45, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 147, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str47 = internal constant [14 x i8] c"genrand_real2\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram48 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str47, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str47, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 154, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str49 = internal constant [14 x i8] c"genrand_real3\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram50 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str49, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str49, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 161, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str51 = internal constant [14 x i8] c"genrand_res53\00", section "llvm.metadata"		; <[14 x i8]*> [#uses=1]
@llvm.dbg.subprogram52 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([14 x i8]* @.str51, i32 0, i32 0), i8* getelementptr ([14 x i8]* @.str51, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 168, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite44 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@llvm.dbg.array57 = internal constant [1 x %0*] [%0* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to %0*)], section "llvm.metadata"		; <[1 x %0*]*> [#uses=1]
@llvm.dbg.composite58 = internal constant %llvm.dbg.composite.type { i32 458773, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 0, i64 0, i64 0, i64 0, i32 0, %0* null, %0* bitcast ([1 x %0*]* @llvm.dbg.array57 to %0*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@.str59 = internal constant [5 x i8] c"main\00", section "llvm.metadata"		; <[5 x i8]*> [#uses=1]
@llvm.dbg.subprogram60 = internal constant %llvm.dbg.subprogram.type { i32 458798, %0* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to %0*), %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i8* getelementptr ([5 x i8]* @.str59, i32 0, i32 0), i8* getelementptr ([5 x i8]* @.str59, i32 0, i32 0), i8* null, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*), i32 175, %0* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite58 to %0*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str69 = internal constant [32 x i8] c"1000 outputs of genrand_int32()\00"		; <[32 x i8]*> [#uses=1]
@.str70 = internal constant [7 x i8] c"%10lu \00"		; <[7 x i8]*> [#uses=1]
@.str71 = internal constant [33 x i8] c"\0A1000 outputs of genrand_real2()\00"		; <[33 x i8]*> [#uses=1]
@.str72 = internal constant [8 x i8] c"%10.8f \00"		; <[8 x i8]*> [#uses=1]

define void @init_genrand(i32 %s) nounwind {
entry:
	tail call void @llvm.dbg.func.start(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %0*))
	tail call void @llvm.dbg.stoppoint(i32 59, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	store i32 %s, i32* getelementptr ([624 x i32]* @mt, i32 0, i32 0), align 32
	tail call void @llvm.dbg.stoppoint(i32 60, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	store i32 1, i32* @mti
	tail call void @llvm.dbg.stoppoint(i32 60, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	br i1 false, label %return, label %bb.nph

bb.nph:		; preds = %entry
	%mti.promoted = load i32* @mti		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%indvar = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb1 ]		; <i32> [#uses=2]
	%mti.tmp.0 = add i32 %indvar, %mti.promoted		; <i32> [#uses=5]
	tail call void @llvm.dbg.stoppoint(i32 61, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%0 = add i32 %mti.tmp.0, -1		; <i32> [#uses=1]
	%1 = getelementptr [624 x i32]* @mt, i32 0, i32 %0		; <i32*> [#uses=1]
	%2 = load i32* %1, align 4		; <i32> [#uses=1]
	%3 = add i32 %mti.tmp.0, -1		; <i32> [#uses=1]
	%4 = getelementptr [624 x i32]* @mt, i32 0, i32 %3		; <i32*> [#uses=1]
	%5 = load i32* %4, align 4		; <i32> [#uses=1]
	%6 = lshr i32 %5, 30		; <i32> [#uses=1]
	%7 = xor i32 %6, %2		; <i32> [#uses=1]
	%8 = mul i32 %7, 1812433253		; <i32> [#uses=1]
	%9 = add i32 %8, %mti.tmp.0		; <i32> [#uses=1]
	%10 = getelementptr [624 x i32]* @mt, i32 0, i32 %mti.tmp.0		; <i32*> [#uses=1]
	store i32 %9, i32* %10, align 4
	tail call void @llvm.dbg.stoppoint(i32 60, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%11 = add i32 %mti.tmp.0, 1		; <i32> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	tail call void @llvm.dbg.stoppoint(i32 60, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	%12 = icmp sgt i32 %11, 623		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %12, label %bb1.return_crit_edge, label %bb

bb1.return_crit_edge:		; preds = %bb1
	store i32 %11, i32* @mti
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	tail call void @llvm.dbg.stoppoint(i32 70, i32 0, %0* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to %0*))
	tail call void @llvm.dbg.region.end(%0* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to %0*))
	ret void
}

declare void @llvm.dbg.func.start(%0*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, %0*) nounwind

declare void @llvm.dbg.region.end(%0*) nounwind

declare i32 @puts(i8* nocapture) nounwind

declare i32 @printf(i8* noalias nocapture, ...) nounwind

declare i32 @putchar(i32) nounwind
