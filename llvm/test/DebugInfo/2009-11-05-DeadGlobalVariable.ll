; RUN: llc %s -o /dev/null
; Here variable bar is optimized away. Do not trip over while trying to generate debug info.


define i32 @foo() nounwind uwtable readnone ssp {
entry:
  ret i32 42, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 139632)", isOptimized: true, emissionKind: 0, file: !17, enums: !1, retainedTypes: !1, subprograms: !3, globals: !12)
!1 = !{i32 0}
!3 = !{!5}
!5 = !MDSubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !17, scope: !6, type: !7, function: i32 ()* @foo)
!6 = !MDFile(filename: "fb.c", directory: "/private/tmp")
!7 = !MDSubroutineType(types: !8)
!8 = !{!9}
!9 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!14}
!14 = !MDGlobalVariable(name: "bar", line: 2, isLocal: true, isDefinition: true, scope: !5, file: !6, type: !9)
!15 = !MDLocation(line: 3, column: 3, scope: !16)
!16 = distinct !MDLexicalBlock(line: 1, column: 11, file: !17, scope: !5)
!17 = !MDFile(filename: "fb.c", directory: "/private/tmp")
!18 = !{i32 1, !"Debug Info Version", i32 3}
