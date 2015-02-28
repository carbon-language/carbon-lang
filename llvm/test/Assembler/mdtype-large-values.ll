; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2}
!named = !{!0, !1, !2}

; CHECK:      !0 = !MDBasicType(name: "name", size: 18446744073709551615, align: 18446744073709551614, encoding: DW_ATE_unsigned_char)
; CHECK-NEXT: !1 = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: !0, size: 18446744073709551615, align: 18446744073709551614, offset: 18446744073709551613)
; CHECK-NEXT: !2 = !MDCompositeType(tag: DW_TAG_array_type, baseType: !0, size: 18446744073709551615, align: 18446744073709551614, offset: 18446744073709551613)
!0 = !MDBasicType(tag: DW_TAG_base_type, name: "name", size: 18446744073709551615, align: 18446744073709551614, encoding: DW_ATE_unsigned_char)
!1 = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: !0, size: 18446744073709551615, align: 18446744073709551614, offset: 18446744073709551613)
!2 = !MDCompositeType(tag: DW_TAG_array_type, baseType: !0, size: 18446744073709551615, align: 18446744073709551614, offset: 18446744073709551613)
