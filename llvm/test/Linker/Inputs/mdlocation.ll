!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}

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
