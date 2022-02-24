; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

@foo = global i32 0

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12}

!0 = !DIFile(filename: "scope.h", directory: "/path/to/dir")
!1 = distinct !{}
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !{}

; CHECK: !5 = !DIGlobalVariable(name: "foo", linkageName: "foo", scope: !0, file: !2, line: 7, type: !3, isLocal: true, isDefinition: false, align: 32)
!5 = !DIGlobalVariable(name: "foo", linkageName: "foo", scope: !0,
                       file: !2, line: 7, type: !3, isLocal: true,
                       isDefinition: false, align: 32)

!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "Class", size: 8, align: 8)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "mem", flags: DIFlagStaticMember, scope: !6, baseType: !3)

; CHECK: !8 = !DIGlobalVariable(name: "mem", scope: !0, type: !9, isLocal: false, isDefinition: true, declaration: !7)
!8 = !DIGlobalVariable(name: "mem", scope: !0, declaration: !7, type: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

; CHECK: !12 = !DIGlobalVariable(name: "bar", linkageName: "bar", scope: !0, file: !2, line: 8, type: !3, isLocal: true, isDefinition: false, templateParams: !11, align: 32)
!10 = !DITemplateTypeParameter(name: "Ty", type: !3)
!11 = distinct !{!10}
!12 = !DIGlobalVariable(name: "bar", linkageName: "bar", scope: !0, file: !2, line: 8, type: !3, isLocal: true, isDefinition: false, templateParams: !11, align: 32)
