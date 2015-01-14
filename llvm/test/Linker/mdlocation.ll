; RUN: llvm-link %s %S/Inputs/mdlocation.ll -o - -S | FileCheck %s

; Test that MDLocations are remapped properly.

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !0, !1, !2, !3, !10, !11, !12, !13, !14, !15}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}

; CHECK:      !0 = !{}
; CHECK-NEXT: !1 = !MDLocation(line: 3, column: 7, scope: !0)
; CHECK-NEXT: !2 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
; CHECK-NEXT: !3 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !2)
; CHECK-NEXT: !4 = distinct !{}
; CHECK-NEXT: !5 = !MDLocation(line: 3, column: 7, scope: !4)
; CHECK-NEXT: !6 = !MDLocation(line: 3, column: 7, scope: !4, inlinedAt: !5)
; CHECK-NEXT: !7 = !MDLocation(line: 3, column: 7, scope: !4, inlinedAt: !6)
; CHECK-NEXT: !8 = distinct !MDLocation(line: 3, column: 7, scope: !0)
; CHECK-NEXT: !9 = distinct !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !8)
; CHECK-NEXT: !10 = distinct !{}
; CHECK-NEXT: !11 = !MDLocation(line: 3, column: 7, scope: !10)
; CHECK-NEXT: !12 = !MDLocation(line: 3, column: 7, scope: !10, inlinedAt: !11)
; CHECK-NEXT: !13 = !MDLocation(line: 3, column: 7, scope: !10, inlinedAt: !12)
; CHECK-NEXT: !14 = distinct !MDLocation(line: 3, column: 7, scope: !0)
; CHECK-NEXT: !15 = distinct !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !14)
!0 = !{} ; Use this as a scope.
!1 = !MDLocation(line: 3, column: 7, scope: !0)
!2 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
!3 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !2)
!4 = distinct !{} ; Test actual remapping.
!5 = !MDLocation(line: 3, column: 7, scope: !4)
!6 = !MDLocation(line: 3, column: 7, scope: !4, inlinedAt: !5)
!7 = !MDLocation(line: 3, column: 7, scope: !4, inlinedAt: !6)
; Test distinct nodes.
!8 = distinct !MDLocation(line: 3, column: 7, scope: !0)
!9 = distinct !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !8)
