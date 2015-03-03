; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin -regalloc=basic < %s | FileCheck %s
; Test to check separate label for inlined function argument.

define i32 @foo(i32 %y) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata i32 %y, i64 0, metadata !0, metadata !MDExpression())
  %0 = tail call i32 (...)* @zoo(i32 %y) nounwind, !dbg !9 ; <i32> [#uses=1]
  ret i32 %0, !dbg !9
}

declare i32 @zoo(...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @bar(i32 %x) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !7, metadata !MDExpression())
  tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !0, metadata !MDExpression()) nounwind
  %0 = tail call i32 (...)* @zoo(i32 1) nounwind, !dbg !12 ; <i32> [#uses=1]
  %1 = add nsw i32 %0, %x, !dbg !13               ; <i32> [#uses=1]
  ret i32 %1, !dbg !13
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!20}

!0 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "y", line: 2, arg: 0, scope: !1, file: !2, type: !6)
!1 = !MDSubprogram(name: "foo", linkageName: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, scopeLine: 2, file: !18, scope: !2, type: !4, function: i32 (i32)* @foo, variables: !15)
!2 = !MDFile(filename: "f.c", directory: "/tmp")
!3 = !MDCompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: 1, file: !18, enums: !19, retainedTypes: !19, subprograms: !17, imports:  null)
!4 = !MDSubroutineType(types: !5)
!5 = !{!6, !6}
!6 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "x", line: 6, arg: 0, scope: !8, file: !2, type: !6)
!8 = !MDSubprogram(name: "bar", linkageName: "bar", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, scopeLine: 6, file: !18, scope: !2, type: !4, function: i32 (i32)* @bar, variables: !16)
!9 = !MDLocation(line: 3, scope: !10)
!10 = distinct !MDLexicalBlock(line: 2, column: 0, file: !18, scope: !1)
!11 = !{i32 1}
!12 = !MDLocation(line: 3, scope: !10, inlinedAt: !13)
!13 = !MDLocation(line: 7, scope: !14)
!14 = distinct !MDLexicalBlock(line: 6, column: 0, file: !18, scope: !8)
!15 = !{!0}
!16 = !{!7}
!17 = !{!1, !8}
!18 = !MDFile(filename: "f.c", directory: "/tmp")
!19 = !{i32 0}

;CHECK: DEBUG_VALUE: bar:x <- E
;CHECK: Ltmp
;CHECK:	DEBUG_VALUE: foo:y <- 1{{$}}
!20 = !{i32 1, !"Debug Info Version", i32 3}
