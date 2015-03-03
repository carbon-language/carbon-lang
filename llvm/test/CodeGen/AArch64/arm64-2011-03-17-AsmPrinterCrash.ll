; RUN: llc < %s -mtriple=arm64-apple-darwin

; rdar://9146594

define void @drt_vsprintf() nounwind ssp {
entry:
  %do_tab_convert = alloca i32, align 4
  br i1 undef, label %if.then24, label %if.else295, !dbg !13

if.then24:                                        ; preds = %entry
  unreachable

if.else295:                                       ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %do_tab_convert, metadata !16, metadata !MDExpression()), !dbg !18
  store i32 0, i32* %do_tab_convert, align 4, !dbg !19
  unreachable
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.gv = !{!0}
!llvm.dbg.sp = !{!1, !7, !10, !11, !12}

!0 = !MDGlobalVariable(name: "vsplive", line: 617, isLocal: true, isDefinition: true, scope: !1, file: !2, type: !6)
!1 = !MDSubprogram(name: "drt_vsprintf", line: 616, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, file: !20, scope: !2, type: !4)
!2 = !MDFile(filename: "print.i", directory: "/Volumes/Ebi/echeng/radars/r9146594")
!3 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (http://llvm.org/git/clang.git git:/git/puzzlebox/clang.git/ c4d1aea01c4444eb81bdbf391f1be309127c3cf1)", isOptimized: true, emissionKind: 0, file: !20, enums: !21, retainedTypes: !21)
!4 = !MDSubroutineType(types: !5)
!5 = !{!6}
!6 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !MDSubprogram(name: "putc_mem", line: 30, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, file: !20, scope: !2, type: !8)
!8 = !MDSubroutineType(types: !9)
!9 = !{null}
!10 = !MDSubprogram(name: "print_double", line: 203, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, file: !20, scope: !2, type: !4)
!11 = !MDSubprogram(name: "print_number", line: 75, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, file: !20, scope: !2, type: !4)
!12 = !MDSubprogram(name: "get_flags", line: 508, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, file: !20, scope: !2, type: !8)
!13 = !MDLocation(line: 653, column: 5, scope: !14)
!14 = distinct !MDLexicalBlock(line: 652, column: 35, file: !20, scope: !15)
!15 = distinct !MDLexicalBlock(line: 616, column: 1, file: !20, scope: !1)
!16 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "do_tab_convert", line: 853, scope: !17, file: !2, type: !6)
!17 = distinct !MDLexicalBlock(line: 850, column: 12, file: !20, scope: !14)
!18 = !MDLocation(line: 853, column: 11, scope: !17)
!19 = !MDLocation(line: 853, column: 29, scope: !17)
!20 = !MDFile(filename: "print.i", directory: "/Volumes/Ebi/echeng/radars/r9146594")
!21 = !{i32 0}
