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

; CHECK-NEXT: !3 = !MDEnumerator(name: "seven", value: 7)
; CHECK-NEXT: !4 = !MDEnumerator(name: "negeight", value: -8)
; CHECK-NEXT: !5 = !MDEnumerator(name: "", value: 0)
!4 = !MDEnumerator(name: "seven", value: 7)
!5 = !MDEnumerator(name: "negeight", value: -8)
!6 = !MDEnumerator(name: "", value: 0)

; CHECK-NEXT: !6 = !MDBasicType(name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
; CHECK-NEXT: !7 = !MDBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
; CHECK-NEXT: !8 = !MDBasicType()
!7 = !MDBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
!8 = !MDBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!9 = !MDBasicType()
!10 = !MDBasicType(tag: DW_TAG_base_type, name: "", size: 0, align: 0, encoding: 0)

; CHECK-NEXT: !9 = distinct !{}
; CHECK-NEXT: !10 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
; CHECK-NEXT: !11 = distinct !{}
; CHECK-NEXT: !12 = !MDFile(filename: "", directory: "")
!11 = distinct !{}
!12 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
!13 = distinct !{}
!14 = !MDFile(filename: "", directory: "")

; CHECK-NEXT: !13 = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 32, align: 32)
!15 = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32, align: 32)

; CHECK-NEXT: !14 = !MDCompositeType(tag: DW_TAG_structure_type, name: "MyType", file: !10, line: 2, size: 32, align: 32, identifier: "MangledMyType")
; CHECK-NEXT: !15 = distinct !MDCompositeType(tag: DW_TAG_structure_type, name: "Base", scope: !14, file: !10, line: 3, size: 128, align: 32, offset: 64, flags: DIFlagPublic, elements: !16, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !15, templateParams: !18, identifier: "MangledBase")
; CHECK-NEXT: !16 = !{!17}
; CHECK-NEXT: !17 = !MDDerivedType(tag: DW_TAG_member, name: "field", scope: !15, file: !10, line: 4, baseType: !6, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
; CHECK-NEXT: !18 = !{!6}
; CHECK-NEXT: !19 = !MDCompositeType(tag: DW_TAG_structure_type, name: "Derived", scope: !14, file: !10, line: 3, baseType: !15, size: 128, align: 32, offset: 64, flags: DIFlagPublic, elements: !20, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !15, templateParams: !18, identifier: "MangledBase")
; CHECK-NEXT: !20 = !{!21}
; CHECK-NEXT: !21 = !MDDerivedType(tag: DW_TAG_inheritance, scope: !19, baseType: !15)
; CHECK-NEXT: !22 = !MDDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !6, size: 32, align: 32, extraData: !15)
; CHECK-NEXT: !23 = !MDCompositeType(tag: DW_TAG_structure_type)
; CHECK-NEXT: !24 = !MDCompositeType(tag: DW_TAG_structure_type, runtimeLang: DW_LANG_Cobol85)
!16 = !MDCompositeType(tag: DW_TAG_structure_type, name: "MyType", file: !12, line: 2, size: 32, align: 32, identifier: "MangledMyType")
!17 = !MDCompositeType(tag: DW_TAG_structure_type, name: "Base", scope: !16, file: !12, line: 3, size: 128, align: 32, offset: 64, flags: DIFlagPublic, elements: !18, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !17, templateParams: !20, identifier: "MangledBase")
!18 = !{!19}
!19 = !MDDerivedType(tag: DW_TAG_member, name: "field", scope: !17, file: !12, line: 4, baseType: !7, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!20 = !{!7}
!21 = !MDCompositeType(tag: DW_TAG_structure_type, name: "Derived", scope: !16, file: !12, line: 3, baseType: !17, size: 128, align: 32, offset: 64, flags: DIFlagPublic, elements: !22, runtimeLang: DW_LANG_C_plus_plus_11, vtableHolder: !17, templateParams: !20, identifier: "MangledBase")
!22 = !{!23}
!23 = !MDDerivedType(tag: DW_TAG_inheritance, scope: !21, baseType: !17)
!24 = !MDDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !7, size: 32, align: 32, extraData: !17)
!25 = !MDCompositeType(tag: DW_TAG_structure_type)
!26 = !MDCompositeType(tag: DW_TAG_structure_type, runtimeLang: 6)

; !25 = !{!7, !7}
; !26 = !MDSubroutineType(flags: DIFlagPublic | DIFlagStaticMember, types: !25)
; !27 = !MDSubroutineType(types: !25)
!27 = !{!7, !7}
!28 = !MDSubroutineType(flags: DIFlagPublic | DIFlagStaticMember, types: !27)
!29 = !MDSubroutineType(flags: 0, types: !27)
!30 = !MDSubroutineType(types: !27)
