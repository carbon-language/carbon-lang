; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; This was pulled from clang's debug-info-template-recursive.cpp test.
; class base { };

; template <class T> class foo : public base  {
;   void operator=(const foo r) { }
; };

; class bar : public foo<void> { };
; bar filters;

; CHECK: DW_TAG_template_type_parameter [{{.*}}]
; CHECK-NEXT: DW_AT_name{{.*}}"T"
; CHECK-NOT: DW_AT_type
; CHECK: NULL

%class.bar = type { i8 }

@filters = global %class.bar zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!36, !37}

!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 (trunk 187958) (llvm/trunk 187964)", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !2, globals: !3, imports: !2)
!1 = !MDFile(filename: "debug-info-template-recursive.cpp", directory: "/usr/local/google/home/echristo/tmp")
!2 = !{}
!3 = !{!4}
!4 = !MDGlobalVariable(name: "filters", line: 10, isLocal: false, isDefinition: true, scope: null, file: !5, type: !6, variable: %class.bar* @filters)
!5 = !MDFile(filename: "debug-info-template-recursive.cpp", directory: "/usr/local/google/home/echristo/tmp")
!6 = !MDCompositeType(tag: DW_TAG_class_type, name: "bar", line: 9, size: 8, align: 8, file: !1, elements: !7)
!7 = !{!8, !31}
!8 = !MDDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !9)
!9 = !MDCompositeType(tag: DW_TAG_class_type, name: "foo<void>", line: 5, size: 8, align: 8, file: !1, elements: !10, templateParams: !29)
!10 = !{!11, !19, !25}
!11 = !MDDerivedType(tag: DW_TAG_inheritance, scope: !9, baseType: !12)
!12 = !MDCompositeType(tag: DW_TAG_class_type, name: "base", line: 3, size: 8, align: 8, file: !1, elements: !13)
!13 = !{!14}
!14 = !MDSubprogram(name: "base", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !12, type: !15, variables: !18)
!15 = !MDSubroutineType(types: !16)
!16 = !{null, !17}
!17 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !12)
!18 = !{i32 786468}
!19 = !MDSubprogram(name: "operator=", linkageName: "_ZN3fooIvEaSES0_", line: 6, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: false, scopeLine: 6, file: !1, scope: !9, type: !20, variables: !24)
!20 = !MDSubroutineType(types: !21)
!21 = !{null, !22, !23}
!22 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !9)
!23 = !MDDerivedType(tag: DW_TAG_const_type, baseType: !9)
!24 = !{i32 786468}
!25 = !MDSubprogram(name: "foo", line: 5, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 5, file: !1, scope: !9, type: !26, variables: !28)
!26 = !MDSubroutineType(types: !27)
!27 = !{null, !22}
!28 = !{i32 786468}
!29 = !{!30}
!30 = !MDTemplateTypeParameter(name: "T", type: null)
!31 = !MDSubprogram(name: "bar", line: 9, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 9, file: !1, scope: !6, type: !32, variables: !35)
!32 = !MDSubroutineType(types: !33)
!33 = !{null, !34}
!34 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !6)
!35 = !{i32 786468}
!36 = !{i32 2, !"Dwarf Version", i32 3}
!37 = !{i32 1, !"Debug Info Version", i32 3}
