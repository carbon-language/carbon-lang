; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!0 = !MDBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!1 = !{null}
!2 = !{null, !0}
!3 = !{!0, !0, !0}


; CHECK: !4 = !MDSubroutineType(types: !1)
; CHECK: !5 = !MDSubroutineType(types: !2)
; CHECK: !6 = !MDSubroutineType(types: !3)
; CHECK: !7 = !MDSubroutineType(flags: DIFlagLValueReference, types: !3)
!4 = !MDSubroutineType(types: !1)
!5 = !MDSubroutineType(types: !2)
!6 = !MDSubroutineType(types: !3)
!7 = !MDSubroutineType(flags: DIFlagLValueReference, types: !3)

; CHECK: !8 = !MDSubroutineType(types: null)
!8 = !MDSubroutineType(types: null)
