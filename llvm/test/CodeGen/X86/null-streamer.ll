; Check the MCNullStreamer operates correctly, at least on a minimal test case.
;
; RUN: llc -filetype=null -o %t -march=x86 %s
; RUN: llc -filetype=null -o %t -mtriple=i686-cygwin %s

define void @f0()  {
  ret void
}

define void @f1() {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !13}

!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: " ", isOptimized: true, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !9, imports: !2)
!1 = !MDFile(filename: "", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !MDSubprogram(name: "", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 2, file: !1, scope: !5, type: !6, function: i32 ()* null, variables: !2)
!5 = !MDFile(filename: "", directory: "")
!6 = !MDSubroutineType(types: !7)
!7 = !{!8}
!8 = !MDBasicType(tag: DW_TAG_base_type, size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !MDGlobalVariable(name: "i", linkageName: "_ZL1i", line: 1, isLocal: true, isDefinition: true, scope: null, file: !5, type: !8)
!11 = !{i32 2, !"Dwarf Version", i32 3}
!13 = !{i32 1, !"Debug Info Version", i32 3}
