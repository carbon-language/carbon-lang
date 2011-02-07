; RUN: opt < %s -simplifycfg -S | FileCheck %s

        %llvm.dbg.anchor.type = type { i32, i32 }
        %llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8* }

@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"

@.str = internal constant [4 x i8] c"a.c\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [6 x i8] c"/tmp/\00", section "llvm.metadata"	; <[6 x i8]*> [#uses=1]
@.str2 = internal constant [55 x i8] c"4.2.1 (Based on Apple Inc. build 5636) (LLVM build 00)\00", section "llvm.metadata"		; <[55 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([55 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

define i1 @t({ i32, i32 }* %I) {
; CHECK: @t
; CHECK: %tmp.2.i.off = add i32 %tmp.2.i, -14
; CHECK: %switch = icmp ult i32 %tmp.2.i.off, 6
entry:
        %tmp.1.i = getelementptr { i32, i32 }* %I, i64 0, i32 1         ; <i32*> [#uses=1]
        %tmp.2.i = load i32* %tmp.1.i           ; <i32> [#uses=6]
        %tmp.2 = icmp eq i32 %tmp.2.i, 14               ; <i1> [#uses=1]
        br i1 %tmp.2, label %shortcirc_done.4, label %shortcirc_next.0
shortcirc_next.0:               ; preds = %entry
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
        %tmp.6 = icmp eq i32 %tmp.2.i, 15               ; <i1> [#uses=1]
        br i1 %tmp.6, label %shortcirc_done.4, label %shortcirc_next.1
shortcirc_next.1:               ; preds = %shortcirc_next.0
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
        %tmp.11 = icmp eq i32 %tmp.2.i, 16              ; <i1> [#uses=1]
        br i1 %tmp.11, label %shortcirc_done.4, label %shortcirc_next.2
shortcirc_next.2:               ; preds = %shortcirc_next.1
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
        %tmp.16 = icmp eq i32 %tmp.2.i, 17              ; <i1> [#uses=1]
        br i1 %tmp.16, label %shortcirc_done.4, label %shortcirc_next.3
shortcirc_next.3:               ; preds = %shortcirc_next.2
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
        %tmp.21 = icmp eq i32 %tmp.2.i, 18              ; <i1> [#uses=1]
        br i1 %tmp.21, label %shortcirc_done.4, label %shortcirc_next.4
shortcirc_next.4:               ; preds = %shortcirc_next.3
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
        %tmp.26 = icmp eq i32 %tmp.2.i, 19              ; <i1> [#uses=1]
        br label %UnifiedReturnBlock
shortcirc_done.4:               ; preds = %shortcirc_next.3, %shortcirc_next.2, %shortcirc_next.1, %shortcirc_next.0, %entry
        br label %UnifiedReturnBlock
UnifiedReturnBlock:             ; preds = %shortcirc_done.4, %shortcirc_next.4
        %UnifiedRetVal = phi i1 [ %tmp.26, %shortcirc_next.4 ], [ true, %shortcirc_done.4 ]             ; <i1> [#uses=1]
        ret i1 %UnifiedRetVal
}

