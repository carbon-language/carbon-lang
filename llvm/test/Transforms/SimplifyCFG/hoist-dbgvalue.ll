; RUN: opt -simplifycfg -S < %s | FileCheck %s

define i32 @foo(i32 %i) nounwind ssp {
  call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata !6, metadata !MDExpression()), !dbg !7
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !9, metadata !MDExpression()), !dbg !11
  %1 = icmp ne i32 %i, 0, !dbg !12
;CHECK: call i32 (...)* @bar()
;CHECK-NEXT: llvm.dbg.value
  br i1 %1, label %2, label %4, !dbg !12

; <label>:2                                       ; preds = %0
  %3 = call i32 (...)* @bar(), !dbg !13
  call void @llvm.dbg.value(metadata i32 %3, i64 0, metadata !9, metadata !MDExpression()), !dbg !13
  br label %6, !dbg !15

; <label>:4                                       ; preds = %0
  %5 = call i32 (...)* @bar(), !dbg !16
  call void @llvm.dbg.value(metadata i32 %5, i64 0, metadata !9, metadata !MDExpression()), !dbg !16
  br label %6, !dbg !18

; <label>:6                                       ; preds = %4, %2
  %k.0 = phi i32 [ %3, %2 ], [ %5, %4 ]
  ret i32 %k.0, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @bar(...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.module.flags = !{!21}
!llvm.dbg.sp = !{!0}

!0 = !MDSubprogram(name: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, file: !20, scope: !1, type: !3, function: i32 (i32)* @foo)
!1 = !MDFile(filename: "b.c", directory: "/private/tmp")
!2 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang", isOptimized: true, emissionKind: 0, file: !20, enums: !8, retainedTypes: !8)
!3 = !MDSubroutineType(types: !4)
!4 = !{!5}
!5 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "i", line: 2, arg: 1, scope: !0, file: !1, type: !5)
!7 = !MDLocation(line: 2, column: 13, scope: !0)
!8 = !{i32 0}
!9 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "k", line: 3, scope: !10, file: !1, type: !5)
!10 = distinct !MDLexicalBlock(line: 2, column: 16, file: !20, scope: !0)
!11 = !MDLocation(line: 3, column: 12, scope: !10)
!12 = !MDLocation(line: 4, column: 3, scope: !10)
!13 = !MDLocation(line: 5, column: 5, scope: !14)
!14 = distinct !MDLexicalBlock(line: 4, column: 10, file: !20, scope: !10)
!15 = !MDLocation(line: 6, column: 3, scope: !14)
!16 = !MDLocation(line: 7, column: 5, scope: !17)
!17 = distinct !MDLexicalBlock(line: 6, column: 10, file: !20, scope: !10)
!18 = !MDLocation(line: 8, column: 3, scope: !17)
!19 = !MDLocation(line: 9, column: 3, scope: !10)
!20 = !MDFile(filename: "b.c", directory: "/private/tmp")
!21 = !{i32 1, !"Debug Info Version", i32 3}
