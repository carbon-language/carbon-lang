!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}

!0 = !MDSubprogram() ; Use this as a scope.
!1 = !MDLocation(line: 3, column: 7, scope: !0)
!2 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
!3 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !2)
!4 = distinct !MDSubprogram() ; Test actual remapping.
!5 = !MDLocation(line: 3, column: 7, scope: !4)
!6 = !MDLocation(line: 3, column: 7, scope: !4, inlinedAt: !5)
!7 = !MDLocation(line: 3, column: 7, scope: !4, inlinedAt: !6)
; Test distinct nodes.
!8 = distinct !MDLocation(line: 3, column: 7, scope: !0)
!9 = distinct !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !8)
