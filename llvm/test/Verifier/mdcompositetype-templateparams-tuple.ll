; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK:      invalid template params
; CHECK-NEXT: !2 = !MDCompositeType(
; CHECK-SAME:                       templateParams: !1
; CHECK-NEXT: !1 = !MDTemplateTypeParameter(

!named = !{!0, !1, !2}
!0 = !MDBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!1 = !MDTemplateTypeParameter(name: "T", type: !0)
!2 = !MDCompositeType(tag: DW_TAG_structure_type, name: "IntTy", size: 32, align: 32, templateParams: !1)
