; RUN: llvm-link %s %S/Inputs/mdlocation.ll -o - -S | FileCheck %s

; Test that MDLocations are remapped properly.

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !0, !1, !2, !3, !8, !9, !10, !11}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7}

; CHECK:      !0 = !{}
; CHECK-NEXT: !1 = !MDLocation(line: 3, column: 7, scope: !0)
; CHECK-NEXT: !2 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
; CHECK-NEXT: !3 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !2)
; CHECK-NEXT: !4 = distinct !{}
; CHECK-NEXT: !5 = !MDLocation(line: 3, column: 7, scope: !4)
; CHECK-NEXT: !6 = !MDLocation(line: 3, column: 7, scope: !4, inlinedAt: !5)
; CHECK-NEXT: !7 = !MDLocation(line: 3, column: 7, scope: !4, inlinedAt: !6)
; CHECK-NEXT: !8 = distinct !{}
; CHECK-NEXT: !9 = !MDLocation(line: 3, column: 7, scope: !8)
; CHECK-NEXT: !10 = !MDLocation(line: 3, column: 7, scope: !8, inlinedAt: !9)
; CHECK-NEXT: !11 = !MDLocation(line: 3, column: 7, scope: !8, inlinedAt: !10)
!0 = !{} ; Use this as a scope.
!1 = !MDLocation(line: 3, column: 7, scope: !0)
!2 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
!3 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !2)
!4 = distinct !{} ; Test actual remapping.
!5 = !MDLocation(line: 3, column: 7, scope: !4)
!6 = !MDLocation(line: 3, column: 7, scope: !4, inlinedAt: !5)
!7 = !MDLocation(line: 3, column: 7, scope: !4, inlinedAt: !6)
