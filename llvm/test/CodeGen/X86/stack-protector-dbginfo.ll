; RUN: llc -mtriple=x86_64-apple-darwin < %s -o -

; PR16954
;
; Make sure that when we splice off the end of a machine basic block, we include
; DBG_VALUE MI in the terminator sequence.

@a = external global { i64, [56 x i8] }, align 32

; Function Attrs: nounwind sspreq
define i32 @_Z18read_response_sizev() #0 {
entry:
  tail call void @llvm.dbg.value(metadata !22, i64 0, metadata !23, metadata !MDExpression()), !dbg !39
  %0 = load i64, i64* getelementptr inbounds ({ i64, [56 x i8] }* @a, i32 0, i32 0), align 8, !dbg !40
  tail call void @llvm.dbg.value(metadata i32 undef, i64 0, metadata !64, metadata !MDExpression()), !dbg !71
  %1 = trunc i64 %0 to i32
  ret i32 %1
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

attributes #0 = { sspreq }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !72}

!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !5, subprograms: !8, globals: !20, imports: !5)
!1 = !MDFile(filename: "<unknown>", directory: "/Users/matt/ryan_bug")
!2 = !{!3}
!3 = !MDCompositeType(tag: DW_TAG_enumeration_type, line: 20, size: 32, align: 32, file: !1, scope: !4, elements: !6)
!4 = !MDCompositeType(tag: DW_TAG_structure_type, name: "C", line: 19, size: 8, align: 8, file: !1, elements: !5)
!5 = !{}
!6 = !{!7}
!7 = !MDEnumerator(name: "max_frame_size", value: 0) ; [ DW_TAG_enumerator ] [max_frame_size :: 0]
!8 = !{!9, !24, !41, !65}
!9 = !MDSubprogram(name: "read_response_size", linkageName: "_Z18read_response_sizev", line: 27, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 27, file: !1, scope: !10, type: !11, function: i32 ()* @_Z18read_response_sizev, variables: !14)
!10 = !MDFile(filename: "<unknown>", directory: "/Users/matt/ryan_bug")
!11 = !MDSubroutineType(types: !12)
!12 = !{!13}
!13 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !{!15, !19}
!15 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "b", line: 28, scope: !9, file: !10, type: !16)
!16 = !MDCompositeType(tag: DW_TAG_structure_type, name: "B", line: 16, size: 32, align: 32, file: !1, elements: !17)
!17 = !{!18}
!18 = !MDDerivedType(tag: DW_TAG_member, name: "end_of_file", line: 17, size: 32, align: 32, file: !1, scope: !16, baseType: !13)
!19 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "c", line: 29, scope: !9, file: !10, type: !13)
!20 = !{}
!21 = !{i32 2, !"Dwarf Version", i32 2}
!22 = !{i64* getelementptr inbounds ({ i64, [56 x i8] }* @a, i32 0, i32 0)}
!23 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "p2", line: 12, arg: 2, scope: !24, file: !10, type: !32, inlinedAt: !38)
!24 = !MDSubprogram(name: "min<unsigned long long>", linkageName: "_ZN3__13minIyEERKT_S3_RS1_", line: 12, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 12, file: !1, scope: !25, type: !27, templateParams: !33, variables: !35)
!25 = !MDNamespace(name: "__1", line: 1, file: !26, scope: null)
!26 = !MDFile(filename: "main.cpp", directory: "/Users/matt/ryan_bug")
!27 = !MDSubroutineType(types: !28)
!28 = !{!29, !29, !32}
!29 = !MDDerivedType(tag: DW_TAG_reference_type, baseType: !30)
!30 = !MDDerivedType(tag: DW_TAG_const_type, baseType: !31)
!31 = !MDBasicType(tag: DW_TAG_base_type, name: "long long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!32 = !MDDerivedType(tag: DW_TAG_reference_type, baseType: !31)
!33 = !{!34}
!34 = !MDTemplateTypeParameter(name: "_Tp", type: !31)
!35 = !{!36, !37}
!36 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "p1", line: 12, arg: 1, scope: !24, file: !10, type: !29)
!37 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "p2", line: 12, arg: 2, scope: !24, file: !10, type: !32)
!38 = !MDLocation(line: 33, scope: !9)
!39 = !MDLocation(line: 12, scope: !24, inlinedAt: !38)
!40 = !MDLocation(line: 9, scope: !41, inlinedAt: !59)
!41 = !MDSubprogram(name: "min<unsigned long long, __1::A>", linkageName: "_ZN3__13minIyNS_1AEEERKT_S4_RS2_T0_", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 8, file: !1, scope: !25, type: !42, templateParams: !53, variables: !55)
!42 = !MDSubroutineType(types: !43)
!43 = !{!29, !29, !32, !44}
!44 = !MDCompositeType(tag: DW_TAG_structure_type, name: "A", size: 8, align: 8, file: !1, scope: !25, elements: !45)
!45 = !{!46}
!46 = !MDSubprogram(name: "operator()", linkageName: "_ZN3__11AclERKiS2_", line: 1, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 1, file: !1, scope: !44, type: !47, variables: !52)
!47 = !MDSubroutineType(types: !48)
!48 = !{!13, !49, !50, !50}
!49 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !44)
!50 = !MDDerivedType(tag: DW_TAG_reference_type, baseType: !51)
!51 = !MDDerivedType(tag: DW_TAG_const_type, baseType: !13)
!52 = !{i32 786468}
!53 = !{!34, !54}
!54 = !MDTemplateTypeParameter(name: "_Compare", type: !44)
!55 = !{!56, !57, !58}
!56 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "p1", line: 7, arg: 1, scope: !41, file: !10, type: !29)
!57 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "p2", line: 7, arg: 2, scope: !41, file: !10, type: !32)
!58 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "p3", line: 8, arg: 3, scope: !41, file: !10, type: !44)
!59 = !MDLocation(line: 13, scope: !24, inlinedAt: !38)
!63 = !{i32 undef}
!64 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "p1", line: 1, arg: 2, scope: !65, file: !10, type: !50, inlinedAt: !40)
!65 = !MDSubprogram(name: "operator()", linkageName: "_ZN3__11AclERKiS2_", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 2, file: !1, scope: !25, type: !47, declaration: !46, variables: !66)
!66 = !{!67, !69, !70}
!67 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !65, type: !68)
!68 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !44)
!69 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "p1", line: 1, arg: 2, scope: !65, file: !10, type: !50)
!70 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "", line: 2, arg: 3, scope: !65, file: !10, type: !50)
!71 = !MDLocation(line: 1, scope: !65, inlinedAt: !40)
!72 = !{i32 1, !"Debug Info Version", i32 3}
