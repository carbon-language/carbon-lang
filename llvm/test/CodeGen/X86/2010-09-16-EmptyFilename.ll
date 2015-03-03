; RUN: llc -O0 -mtriple=x86_64-apple-darwin10 < %s - | FileCheck %s
; Radar 8286101
; CHECK: .file   {{[0-9]+}} "<stdin>"

define i32 @foo() nounwind ssp {
entry:
  ret i32 42, !dbg !8
}

define i32 @bar() nounwind ssp {
entry:
  ret i32 21, !dbg !10
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17}

!0 = !MDSubprogram(name: "foo", linkageName: "foo", line: 53, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !14, scope: !1, type: !3, function: i32 ()* @foo)
!1 = !MDFile(filename: "", directory: "/private/tmp")
!2 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 114084)", isOptimized: false, emissionKind: 0, file: !15, enums: !16, retainedTypes: !16, subprograms: !13)
!3 = !MDSubroutineType(types: !4)
!4 = !{!5}
!5 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !MDSubprogram(name: "bar", linkageName: "bar", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !15, scope: !7, type: !3, function: i32 ()* @bar)
!7 = !MDFile(filename: "bug.c", directory: "/private/tmp")
!8 = !MDLocation(line: 53, column: 13, scope: !9)
!9 = distinct !MDLexicalBlock(line: 53, column: 11, file: !14, scope: !0)
!10 = !MDLocation(line: 4, column: 13, scope: !11)
!11 = distinct !MDLexicalBlock(line: 4, column: 13, file: !15, scope: !12)
!12 = distinct !MDLexicalBlock(line: 4, column: 11, file: !15, scope: !6)
!13 = !{!0, !6}
!14 = !MDFile(filename: "", directory: "/private/tmp")
!15 = !MDFile(filename: "bug.c", directory: "/private/tmp")
!16 = !{i32 0}
!17 = !{i32 1, !"Debug Info Version", i32 3}
