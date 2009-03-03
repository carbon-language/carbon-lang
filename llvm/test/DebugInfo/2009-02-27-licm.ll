;RUN: llvm-as <%s | opt -licm | llvm-dis | grep {load } | count 4
; ModuleID = '2009-02-27-licm.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i32 }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [12 x i8] c"mt19937ar.c\00", section "llvm.metadata"		; <[12 x i8]*> [#uses=1]
@.str1 = internal constant [58 x i8] c"/developer2/home5/youxiangc/work/project/pr965/test-licm/\00", section "llvm.metadata"		; <[58 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5641) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([58 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null, i32 0 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@mti = internal global i32 625		; <i32*> [#uses=7]
@.str5 = internal constant [18 x i8] c"long unsigned int\00", section "llvm.metadata"		; <[18 x i8]*> [#uses=1]
@llvm.dbg.basictype6 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 7 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array = internal constant [2 x { }*] [{ }* null, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype6 to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str7 = internal constant [13 x i8] c"init_genrand\00", section "llvm.metadata"		; <[13 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([13 x i8]* @.str7, i32 0, i32 0), i8* getelementptr ([13 x i8]* @.str7, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 13, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@mt = internal global [624 x i32] zeroinitializer, align 32		; <[624 x i32]*> [#uses=4]

define void @init_genrand(i32 %s) nounwind {
entry:
	tail call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	tail call void @llvm.dbg.stoppoint(i32 14, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i32 %s, i32* getelementptr ([624 x i32]* @mt, i32 0, i32 0), align 32
	tail call void @llvm.dbg.stoppoint(i32 15, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	store i32 1, i32* @mti
	tail call void @llvm.dbg.stoppoint(i32 15, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%0 = load i32* @mti, align 4		; <i32> [#uses=1]
	%1 = icmp sgt i32 %0, 623		; <i1> [#uses=1]
	br i1 %1, label %return, label %bb.nph

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%storemerge1 = phi i32 [ %16, %bb1 ], [ 1, %bb.nph ]		; <i32> [#uses=0]
	tail call void @llvm.dbg.stoppoint(i32 16, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%2 = load i32* @mti, align 4		; <i32> [#uses=3]
	%3 = add i32 %2, -1		; <i32> [#uses=1]
	%4 = getelementptr [624 x i32]* @mt, i32 0, i32 %3		; <i32*> [#uses=1]
	%5 = load i32* %4, align 4		; <i32> [#uses=1]
	%6 = add i32 %2, -1		; <i32> [#uses=1]
	%7 = getelementptr [624 x i32]* @mt, i32 0, i32 %6		; <i32*> [#uses=1]
	%8 = load i32* %7, align 4		; <i32> [#uses=1]
	%9 = lshr i32 %8, 30		; <i32> [#uses=1]
	%10 = xor i32 %9, %5		; <i32> [#uses=1]
	%11 = mul i32 %10, 1812433253		; <i32> [#uses=1]
	%12 = load i32* @mti, align 4		; <i32> [#uses=1]
	%13 = add i32 %11, %12		; <i32> [#uses=1]
	%14 = getelementptr [624 x i32]* @mt, i32 0, i32 %2		; <i32*> [#uses=1]
	store i32 %13, i32* %14, align 4
	tail call void @llvm.dbg.stoppoint(i32 15, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%15 = load i32* @mti, align 4		; <i32> [#uses=1]
	%16 = add i32 %15, 1		; <i32> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%storemerge = phi i32 [ %16, %bb ]		; <i32> [#uses=1]
	store i32 %storemerge, i32* @mti
	tail call void @llvm.dbg.stoppoint(i32 15, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	%17 = load i32* @mti, align 4		; <i32> [#uses=1]
	%18 = icmp sgt i32 %17, 623		; <i1> [#uses=1]
	br i1 %18, label %bb1.return_crit_edge, label %bb

bb1.return_crit_edge:		; preds = %bb1
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	tail call void @llvm.dbg.stoppoint(i32 25, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	tail call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	ret void
}

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

declare void @llvm.dbg.region.end({ }*) nounwind
