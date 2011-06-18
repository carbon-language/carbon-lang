; RUN: opt < %s -simplifycfg -S | \
; RUN:   not grep " switch"

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i32 }
	%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
	%llvm.dbg.variable.type = type { i32, { }*, i8*, { }*, i32, { }* }
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [10 x i8] c"swithh2.c\00", section "llvm.metadata"		; <[10 x i8]*> [#uses=1]
@.str1 = internal constant [38 x i8] c"/developer/home2/zsth/test/debug/tmp/\00", section "llvm.metadata"		; <[38 x i8]*> [#uses=1]
@.str2 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5641) (LLVM build)\00", section "llvm.metadata"		; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([10 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([38 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null, i32 -1 }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@llvm.dbg.array = internal constant [2 x { }*] [{ }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*)], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0, i64 0, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array to { }*), i32 0 }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 46 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str4 = internal constant [4 x i8] c"foo\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.subprograms to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str4, i32 0, i32 0), i8* getelementptr ([4 x i8]* @.str4, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 1, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite to { }*), i1 false, i1 true }, section "llvm.metadata"		; <%llvm.dbg.subprogram.type*> [#uses=1]
@.str5 = internal constant [2 x i8] c"x\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.variable = internal constant %llvm.dbg.variable.type { i32 459009, { }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*), i8* getelementptr ([2 x i8]* @.str5, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 1, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*) }, section "llvm.metadata"		; <%llvm.dbg.variable.type*> [#uses=0]

define i32 @foo(i32 %x) nounwind {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.dbg.func.start({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	call void @llvm.dbg.stoppoint(i32 2, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	switch i32 %x, label %bb4 [
		i32 1, label %bb
		i32 2, label %bb1
		i32 3, label %bb2
		i32 4, label %bb3
	]
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 2, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb

bb:		; preds = %0, %entry
	call void @llvm.dbg.stoppoint(i32 3, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb8
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 3, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb1

bb1:		; preds = %1, %entry
	call void @llvm.dbg.stoppoint(i32 4, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb8
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 4, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb2

bb2:		; preds = %2, %entry
	call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb8
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb3

bb3:		; preds = %3, %entry
	call void @llvm.dbg.stoppoint(i32 6, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb8
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 6, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb4

bb4:		; preds = %4, %entry
	call void @llvm.dbg.stoppoint(i32 10, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	switch i32 %x, label %bb7 [
		i32 5, label %bb5
		i32 6, label %bb6
	]
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 10, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb5

bb5:		; preds = %5, %bb4
	call void @llvm.dbg.stoppoint(i32 11, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb8
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 11, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb6

bb6:		; preds = %6, %bb4
	call void @llvm.dbg.stoppoint(i32 12, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb8
		; No predecessors!
	call void @llvm.dbg.stoppoint(i32 12, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb7

bb7:		; preds = %7, %bb4
	call void @llvm.dbg.stoppoint(i32 13, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %bb8

bb8:		; preds = %bb7, %bb6, %bb5, %bb3, %bb2, %bb1, %bb
	%.0 = phi i32 [ 4, %bb3 ], [ 3, %bb2 ], [ 2, %bb1 ], [ 1, %bb ], [ 6, %bb6 ], [ 5, %bb5 ], [ 0, %bb7 ]		; <i32> [#uses=1]
	call void @llvm.dbg.stoppoint(i32 13, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	br label %return

return:		; preds = %bb8
	call void @llvm.dbg.stoppoint(i32 13, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
	call void @llvm.dbg.region.end({ }* bitcast (%llvm.dbg.subprogram.type* @llvm.dbg.subprogram to { }*))
	ret i32 %.0
}

declare void @llvm.dbg.func.start({ }*) nounwind

declare void @llvm.dbg.declare({ }*, { }*) nounwind

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

declare void @llvm.dbg.region.end({ }*) nounwind
