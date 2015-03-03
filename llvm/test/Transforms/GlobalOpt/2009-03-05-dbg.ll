; REQUIRES: asserts
; RUN: opt < %s -globalopt -stats -disable-output 2>&1 | grep "1 globalopt - Number of global vars shrunk to booleans"

@Stop = internal global i32 0                     ; <i32*> [#uses=3]

define i32 @foo(i32 %i) nounwind ssp {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata !3, metadata !{})
  %0 = icmp eq i32 %i, 1, !dbg !7                 ; <i1> [#uses=1]
  br i1 %0, label %bb, label %bb1, !dbg !7

bb:                                               ; preds = %entry
  store i32 0, i32* @Stop, align 4, !dbg !9
  %1 = mul nsw i32 %i, 42, !dbg !10               ; <i32> [#uses=1]
  call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !3, metadata !{}), !dbg !10
  br label %bb2, !dbg !10

bb1:                                              ; preds = %entry
  store i32 1, i32* @Stop, align 4, !dbg !11
  br label %bb2, !dbg !11

bb2:                                              ; preds = %bb1, %bb
  %i_addr.0 = phi i32 [ %1, %bb ], [ %i, %bb1 ]   ; <i32> [#uses=1]
  br label %return, !dbg !12

return:                                           ; preds = %bb2
  ret i32 %i_addr.0, !dbg !12
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @bar() nounwind ssp {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load i32, i32* @Stop, align 4, !dbg !13         ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 1, !dbg !13                ; <i1> [#uses=1]
  br i1 %1, label %bb, label %bb1, !dbg !13

bb:                                               ; preds = %entry
  br label %bb2, !dbg !18

bb1:                                              ; preds = %entry
  br label %bb2, !dbg !19

bb2:                                              ; preds = %bb1, %bb
  %.0 = phi i32 [ 0, %bb ], [ 1, %bb1 ]           ; <i32> [#uses=1]
  br label %return, !dbg !19

return:                                           ; preds = %bb2
  ret i32 %.0, !dbg !19
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.gv = !{!0}

!0 = !MDGlobalVariable(name: "Stop", line: 2, isLocal: true, isDefinition: true, scope: !1, file: !1, type: !2, variable: i32* @Stop)
!1 = !MDCompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: 0, file: !20, enums: !21, retainedTypes: !21)
!2 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!3 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "i", line: 4, arg: 0, scope: !4, file: !1, type: !2)
!4 = !MDSubprogram(name: "foo", linkageName: "foo", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scope: !1, type: !5)
!5 = !MDSubroutineType(types: !6)
!6 = !{!2, !2}
!7 = !MDLocation(line: 5, scope: !8)
!8 = distinct !MDLexicalBlock(line: 0, column: 0, file: !20, scope: !4)
!9 = !MDLocation(line: 6, scope: !8)
!10 = !MDLocation(line: 7, scope: !8)
!11 = !MDLocation(line: 9, scope: !8)
!12 = !MDLocation(line: 11, scope: !8)
!13 = !MDLocation(line: 14, scope: !14)
!14 = distinct !MDLexicalBlock(line: 0, column: 0, file: !20, scope: !15)
!15 = !MDSubprogram(name: "bar", linkageName: "bar", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scope: !1, type: !16)
!16 = !MDSubroutineType(types: !17)
!17 = !{!2}
!18 = !MDLocation(line: 15, scope: !14)
!19 = !MDLocation(line: 16, scope: !14)
!20 = !MDFile(filename: "g.c", directory: "/tmp")
!21 = !{i32 0}
