; RUN: llc < %s -o /dev/null
; PR 5942
define i8* @foo() nounwind {
entry:
  %0 = load i32, i32* undef, align 4, !dbg !0          ; <i32> [#uses=1]
  %1 = inttoptr i32 %0 to i8*, !dbg !0            ; <i8*> [#uses=1]
  ret i8* %1, !dbg !10

}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!14}

!0 = !MDLocation(line: 571, column: 3, scope: !1)
!1 = distinct !MDLexicalBlock(line: 1, column: 1, file: !11, scope: !2)
!2 = !MDSubprogram(name: "foo", linkageName: "foo", line: 561, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scope: !3, type: !4)
!3 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang 1.1", isOptimized: true, emissionKind: 0, file: !11, enums: !12, retainedTypes: !12, subprograms: !13)
!4 = !MDSubroutineType(types: !5)
!5 = !{!6}
!6 = !MDBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!10 = !MDLocation(line: 588, column: 1, scope: !2)
!11 = !MDFile(filename: "hashtab.c", directory: "/usr/src/gnu/usr.bin/cc/cc_tools/../../../../contrib/gcclibs/libiberty")
!12 = !{i32 0}
!13 = !{!2}
!14 = !{i32 1, !"Debug Info Version", i32 3}
