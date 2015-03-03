; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Check that two compile units are generated

; CHECK: Compile Unit:
; CHECK: Compile Unit:

define i32 @foo() nounwind readnone ssp {
return:
  ret i32 42, !dbg !0
}

define i32 @bar() nounwind readnone ssp {
return:
  ret i32 21, !dbg !8
}

!llvm.dbg.cu = !{!4, !12}
!llvm.module.flags = !{!21}
!16 = !{!2}
!17 = !{!10}

!0 = !MDLocation(line: 3, scope: !1)
!1 = distinct !MDLexicalBlock(line: 2, column: 0, file: !18, scope: !2)
!2 = !MDSubprogram(name: "foo", linkageName: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !18, scope: !3, type: !5, function: i32 ()* @foo)
!3 = !MDFile(filename: "a.c", directory: "/tmp/")
!4 = !MDCompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: false, emissionKind: 0, file: !18, enums: !19, retainedTypes: !19, subprograms: !16)
!5 = !MDSubroutineType(types: !6)
!6 = !{!7}
!7 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !MDLocation(line: 3, scope: !9)
!9 = distinct !MDLexicalBlock(line: 2, column: 0, file: !20, scope: !10)
!10 = !MDSubprogram(name: "bar", linkageName: "bar", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !20, scope: !11, type: !13, function: i32 ()* @bar)
!11 = !MDFile(filename: "b.c", directory: "/tmp/")
!12 = !MDCompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: false, emissionKind: 0, file: !20, enums: !19, retainedTypes: !19, subprograms: !17)
!13 = !MDSubroutineType(types: !14)
!14 = !{!15}
!15 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!18 = !MDFile(filename: "a.c", directory: "/tmp/")
!19 = !{i32 0}
!20 = !MDFile(filename: "b.c", directory: "/tmp/")
!21 = !{i32 1, !"Debug Info Version", i32 3}
