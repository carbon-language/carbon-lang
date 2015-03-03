; RUN: opt -strip-dead-debug-info -disable-output < %s
define i32 @foo() nounwind ssp {
entry:
  ret i32 0, !dbg !8
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14}

!0 = !MDSubprogram(name: "foo", linkageName: "foo", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !10, scope: !1, type: !3, function: i32 ()* @foo)
!1 = !MDFile(filename: "/tmp/a.c", directory: "/Volumes/Lalgate/clean/D.CW")
!2 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 2.8 (trunk 112062)", isOptimized: true, emissionKind: 1, file: !10, enums: !11, retainedTypes: !11, subprograms: !12, globals: !13)
!3 = !MDSubroutineType(types: !4)
!4 = !{!5}
!5 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !MDGlobalVariable(name: "i", linkageName: "i", line: 2, isLocal: true, isDefinition: true, scope: !1, file: !1, type: !7)
!7 = !MDDerivedType(tag: DW_TAG_const_type, file: !10, scope: !1, baseType: !5)
!8 = !MDLocation(line: 3, column: 13, scope: !9)
!9 = distinct !MDLexicalBlock(line: 3, column: 11, file: !10, scope: !0)
!10 = !MDFile(filename: "/tmp/a.c", directory: "/Volumes/Lalgate/clean/D.CW")
!11 = !{i32 0}
!12 = !{!0}
!13 = !{!6}
!14 = !{i32 1, !"Debug Info Version", i32 3}
