; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !0, !1, !2, !3, !4, !5, !6, !7, !8, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !27}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30}

; CHECK:      !0 = !MDSubrange(count: 3)
; CHECK-NEXT: !1 = !MDSubrange(count: 3, lowerBound: 4)
; CHECK-NEXT: !2 = !MDSubrange(count: 3, lowerBound: -5)
!0 = !MDSubrange(count: 3)
!1 = !MDSubrange(count: 3, lowerBound: 0)

!2 = !MDSubrange(count: 3, lowerBound: 4)
!3 = !MDSubrange(count: 3, lowerBound: -5)

; CHECK-NEXT: !3 = !MDEnumerator(value: 7, name: "seven")
; CHECK-NEXT: !4 = !MDEnumerator(value: -8, name: "negeight")
; CHECK-NEXT: !5 = !MDEnumerator(value: 0, name: "")
!4 = !MDEnumerator(value: 7, name: "seven")
!5 = !MDEnumerator(value: -8, name: "negeight")
!6 = !MDEnumerator(value: 0, name: "")

; CHECK-NEXT: !6 = !MDBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
; CHECK-NEXT: !7 = !MDBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
; CHECK-NEXT: !8 = !MDBasicType(tag: DW_TAG_base_type)
!7 = !MDBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
!8 = !MDBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!9 = !MDBasicType(tag: DW_TAG_base_type)
!10 = !MDBasicType(tag: DW_TAG_base_type, name: "", size: 0, align: 0, encoding: 0)

; CHECK-NEXT: !9 = !{!"path/to/file", !"/path/to/dir"}
; CHECK-NEXT: !10 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
; CHECK-NEXT: !11 = !{null, null}
; CHECK-NEXT: !12 = !MDFile(filename: "", directory: "")
!11 = !{!"path/to/file", !"/path/to/dir"}
!12 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
!13 = !{null, null}
!14 = !MDFile(filename: "", directory: "")

; CHECK-NEXT: !13 = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 32, align: 32)
!15 = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32, align: 32)

; CHECK-NEXT: !14 = !MDCompositeType(tag: DW_TAG_structure_type, name: "MyType", file: !9, line: 2, size: 32, align: 32, identifier: "MangledMyType")
; CHECK-NEXT: !15 = distinct !MDCompositeType(tag: DW_TAG_structure_type, name: "Base", file: !9, line: 3, scope: !14, size: 128, align: 32, offset: 64, flags: 3, elements: !16, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !15, templateParams: !18, identifier: "MangledBase")
; CHECK-NEXT: !16 = !{!17}
; CHECK-NEXT: !17 = !MDDerivedType(tag: DW_TAG_member, name: "field", file: !9, line: 4, scope: !15, baseType: !6, size: 32, align: 32, offset: 32, flags: 3)
; CHECK-NEXT: !18 = !{!6}
; CHECK-NEXT: !19 = !MDCompositeType(tag: DW_TAG_structure_type, name: "Derived", file: !9, line: 3, scope: !14, baseType: !15, size: 128, align: 32, offset: 64, flags: 3, elements: !20, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !15, templateParams: !18, identifier: "MangledBase")
; CHECK-NEXT: !20 = !{!21}
; CHECK-NEXT: !21 = !MDDerivedType(tag: DW_TAG_inheritance, scope: !19, baseType: !15)
; CHECK-NEXT: !22 = !MDDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !6, size: 32, align: 32, extraData: !15)
; CHECK-NEXT: !23 = !MDCompositeType(tag: DW_TAG_structure_type)
; CHECK-NEXT: !24 = !MDCompositeType(tag: DW_TAG_structure_type, runtimeLang: DW_LANG_Cobol85)
!16 = !MDCompositeType(tag: DW_TAG_structure_type, name: "MyType", file: !11, line: 2, size: 32, align: 32, identifier: "MangledMyType")
!17 = !MDCompositeType(tag: DW_TAG_structure_type, name: "Base", file: !11, line: 3, scope: !16, size: 128, align: 32, offset: 64, flags: 3, elements: !18, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !17, templateParams: !20, identifier: "MangledBase")
!18 = !{!19}
!19 = !MDDerivedType(tag: DW_TAG_member, name: "field", file: !11, line: 4, scope: !17, baseType: !7, size: 32, align: 32, offset: 32, flags: 3)
!20 = !{!7}
!21 = !MDCompositeType(tag: DW_TAG_structure_type, name: "Derived", file: !11, line: 3, scope: !16, baseType: !17, size: 128, align: 32, offset: 64, flags: 3, elements: !22, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !17, templateParams: !20, identifier: "MangledBase")
!22 = !{!23}
!23 = !MDDerivedType(tag: DW_TAG_inheritance, scope: !21, baseType: !17)
!24 = !MDDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !7, size: 32, align: 32, extraData: !17)
!25 = !MDCompositeType(tag: DW_TAG_structure_type)
!26 = !MDCompositeType(tag: DW_TAG_structure_type, runtimeLang: 6)

; !25 = !{!7, !7}
; !26 = !MDSubroutineType(flags: 7, types: !25)
; !27 = !MDSubroutineType(types: !25)
!27 = !{!7, !7}
!28 = !MDSubroutineType(flags: 7, types: !27)
!29 = !MDSubroutineType(flags: 0, types: !27)
!30 = !MDSubroutineType(types: !27)
