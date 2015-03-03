; RUN: llc < %s -o /dev/null

define void @baz(i32 %i) nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata !0, metadata !1, metadata !MDExpression()), !dbg !0
  ret void, !dbg !0
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!22}

!0 = !{{ [0 x i8] }** undef}
!1 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "x", line: 11, scope: !2, file: !4, type: !9)
!2 = distinct !MDLexicalBlock(line: 8, column: 0, file: !20, scope: !3)
!3 = !MDSubprogram(name: "baz", linkageName: "baz", line: 8, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !20, scope: null, type: !6)
!4 = !MDFile(filename: "2007-12-VarArrayDebug.c", directory: "/Users/sabre/llvm/test/FrontendC/")
!5 = !MDCompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: 0, file: !20, enums: !21, retainedTypes: !21)
!6 = !MDSubroutineType(types: !7)
!7 = !{null, !8}
!8 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !20, scope: !4, baseType: !10)
!10 = !MDCompositeType(tag: DW_TAG_structure_type, line: 11, size: 8, align: 8, file: !20, scope: !3, elements: !11)
!11 = !{!12}
!12 = !MDDerivedType(tag: DW_TAG_member, name: "b", line: 11, size: 8, align: 8, file: !20, scope: !10, baseType: !13)
!13 = !MDDerivedType(tag: DW_TAG_typedef, name: "A", line: 11, file: !20, scope: !3, baseType: !14)
!14 = !MDCompositeType(tag: DW_TAG_array_type, size: 8, align: 8, file: !20, scope: !4, baseType: !15, elements: !16)
!15 = !MDBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!16 = !{!17}
!17 = !MDSubrange(count: 1)
!18 = !{!"llvm.mdnode.fwdref.19"}
!19 = !{!"llvm.mdnode.fwdref.23"}
!20 = !MDFile(filename: "2007-12-VarArrayDebug.c", directory: "/Users/sabre/llvm/test/FrontendC/")
!21 = !{i32 0}
!22 = !{i32 1, !"Debug Info Version", i32 3}
