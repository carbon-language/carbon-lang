; RUN: opt -strip-debug < %s -S | FileCheck %s

; CHECK-NOT: llvm.dbg

@x = common global i32 0                          ; <i32*> [#uses=0]

define void @foo() nounwind readnone optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !5, metadata !{}), !dbg !10
  ret void, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13}
!llvm.dbg.sp = !{!0}
!llvm.dbg.lv.foo = !{!5}
!llvm.dbg.gv = !{!8}

!0 = !MDSubprogram(name: "foo", linkageName: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !12, scope: !1, type: !3, function: void ()* @foo)
!1 = !MDFile(filename: "b.c", directory: "/tmp")
!2 = !MDCompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: 0, file: !12, enums: !4, retainedTypes: !4)
!3 = !MDSubroutineType(types: !4)
!4 = !{null}
!5 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "y", line: 3, scope: !6, file: !1, type: !7)
!6 = distinct !MDLexicalBlock(line: 2, column: 0, file: !12, scope: !0)
!7 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !MDGlobalVariable(name: "x", line: 1, isLocal: false, isDefinition: true, scope: !1, file: !1, type: !7, variable: i32* @x)
!9 = !{i32 0}
!10 = !MDLocation(line: 3, scope: !6)
!11 = !MDLocation(line: 4, scope: !6)
!12 = !MDFile(filename: "b.c", directory: "/tmp")
!13 = !{i32 1, !"Debug Info Version", i32 3}
