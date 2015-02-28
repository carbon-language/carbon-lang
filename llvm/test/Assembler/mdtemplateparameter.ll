; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !3, !4, !5, !5, !6}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!0 = distinct !{}
!1 = distinct !{}
; CHECK: !1 = distinct !{}

; CHECK-NEXT: !2 = !MDTemplateTypeParameter(name: "Ty", type: !1)
; CHECK-NEXT: !3 = !MDTemplateTypeParameter(type: !1)
!2 = !MDTemplateTypeParameter(name: "Ty", type: !1)
!3 = !MDTemplateTypeParameter(type: !1)
!4 = !MDTemplateTypeParameter(name: "", type: !1)

; CHECK-NEXT: !4 = !MDTemplateValueParameter(name: "V", type: !1, value: i32 7)
; CHECK-NEXT: !5 = !MDTemplateValueParameter(type: !1, value: i32 7)
; CHECK-NEXT: !6 = !MDTemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "param", type: !1, value: !"template")
!5 = !MDTemplateValueParameter(name: "V", type: !1, value: i32 7)
!6 = !MDTemplateValueParameter(type: !1, value: i32 7)
!7 = !MDTemplateValueParameter(tag: DW_TAG_template_value_parameter,
                               name: "", type: !1, value: i32 7)
!8 = !MDTemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "param", type: !1, value: !"template")
