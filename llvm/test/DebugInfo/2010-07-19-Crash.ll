; RUN: llc -o /dev/null < %s
; PR7662
; Do not add variables to !11 because it is a declaration entry.

define i32 @bar() nounwind readnone ssp {
entry:
  ret i32 42, !dbg !9
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15}
!llvm.dbg.sp = !{!0, !6, !11}
!llvm.dbg.lv.foo = !{!7}

!0 = !MDSubprogram(name: "bar", linkageName: "bar", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !12, scope: !1, type: !3, function: i32 ()* @bar)
!1 = !MDFile(filename: "one.c", directory: "/private/tmp")
!2 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang 2.8", isOptimized: true, emissionKind: 0, file: !12, enums: !14, retainedTypes: !14, subprograms: !13)
!3 = !MDSubroutineType(types: !4)
!4 = !{!5}
!5 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !MDSubprogram(name: "foo", linkageName: "foo", line: 7, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !12, scope: !1, type: !3)
!7 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "one", line: 8, scope: !8, file: !1, type: !5)
!8 = distinct !MDLexicalBlock(line: 7, column: 18, file: !12, scope: !6)
!9 = !MDLocation(line: 4, column: 3, scope: !10)
!10 = distinct !MDLexicalBlock(line: 3, column: 11, file: !12, scope: !0)
!11 = !MDSubprogram(name: "foo", linkageName: "foo", line: 7, isLocal: true, isDefinition: false, virtualIndex: 6, isOptimized: true, file: !12, scope: !1, type: !3)
!12 = !MDFile(filename: "one.c", directory: "/private/tmp")
!13 = !{!0}
!14 = !{i32 0}
!15 = !{i32 1, !"Debug Info Version", i32 3}
