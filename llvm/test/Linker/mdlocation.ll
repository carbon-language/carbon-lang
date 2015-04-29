; RUN: llvm-link %s %S/Inputs/mdlocation.ll -o - -S | FileCheck %s

; Test that DILocations are remapped properly.

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !0, !1, !2, !3, !10, !11, !12, !13, !14, !15}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}

; CHECK:      !0 = !DISubprogram(
; CHECK-NEXT: !1 = !DILocation(line: 3, column: 7, scope: !0)
; CHECK-NEXT: !2 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
; CHECK-NEXT: !3 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !2)
; CHECK-NEXT: !4 = distinct !DISubprogram(
; CHECK-NEXT: !5 = !DILocation(line: 3, column: 7, scope: !4)
; CHECK-NEXT: !6 = !DILocation(line: 3, column: 7, scope: !4, inlinedAt: !5)
; CHECK-NEXT: !7 = !DILocation(line: 3, column: 7, scope: !4, inlinedAt: !6)
; CHECK-NEXT: !8 = distinct !DILocation(line: 3, column: 7, scope: !0)
; CHECK-NEXT: !9 = distinct !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !8)
; CHECK-NEXT: !10 = distinct !DISubprogram(
; CHECK-NEXT: !11 = !DILocation(line: 3, column: 7, scope: !10)
; CHECK-NEXT: !12 = !DILocation(line: 3, column: 7, scope: !10, inlinedAt: !11)
; CHECK-NEXT: !13 = !DILocation(line: 3, column: 7, scope: !10, inlinedAt: !12)
; CHECK-NEXT: !14 = distinct !DILocation(line: 3, column: 7, scope: !0)
; CHECK-NEXT: !15 = distinct !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !14)
!0 = !DISubprogram() ; Use this as a scope.
!1 = !DILocation(line: 3, column: 7, scope: !0)
!2 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
!3 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !2)
!4 = distinct !DISubprogram() ; Test actual remapping.
!5 = !DILocation(line: 3, column: 7, scope: !4)
!6 = !DILocation(line: 3, column: 7, scope: !4, inlinedAt: !5)
!7 = !DILocation(line: 3, column: 7, scope: !4, inlinedAt: !6)
; Test distinct nodes.
!8 = distinct !DILocation(line: 3, column: 7, scope: !0)
!9 = distinct !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !8)
