; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !3}
!named = !{!0, !1, !2, !3, !4}

; CHECK:      !0 = !MDSubprogram({{.*}})
; CHECK-NEXT: !1 = !MDCompositeType({{.*}})
!0 = !MDSubprogram(name: "foo")
!1 = !MDCompositeType(tag: DW_TAG_structure_type, name: "Class", size: 32, align: 32)

; CHECK-NEXT: !2 = !MDImportedEntity(tag: DW_TAG_imported_module, name: "foo", scope: !0, entity: !1, line: 7)
!2 = !MDImportedEntity(tag: DW_TAG_imported_module, name: "foo", scope: !0,
                       entity: !1, line: 7)

; CHECK-NEXT: !3 = !MDImportedEntity(tag: DW_TAG_imported_module, scope: !0)
!3 = !MDImportedEntity(tag: DW_TAG_imported_module, scope: !0)
!4 = !MDImportedEntity(tag: DW_TAG_imported_module, name: "", scope: !0, entity: null,
                       line: 0)

