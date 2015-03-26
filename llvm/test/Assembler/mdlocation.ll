; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !1, !2, !2, !3, !3, !4}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7}

; CHECK: !0 = !MDSubprogram(
!0 = !MDSubprogram()

; CHECK-NEXT: !1 = !MDLocation(line: 3, column: 7, scope: !0)
!1 = !MDLocation(line: 3, column: 7, scope: !0)
!2 = !MDLocation(scope: !0, column: 7, line: 3)

; CHECK-NEXT: !2 = !MDLocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
!3 = !MDLocation(scope: !0, inlinedAt: !1, column: 7, line: 3)
!4 = !MDLocation(column: 7, line: 3, scope: !0, inlinedAt: !1)

; CHECK-NEXT: !3 = !MDLocation(line: 0, scope: !0)
!5 = !MDLocation(scope: !0)
!6 = !MDLocation(scope: !0, column: 0, line: 0)

; CHECK-NEXT: !4 = !MDLocation(line: 4294967295, column: 65535, scope: !0)
!7 = !MDLocation(line: 4294967295, column: 65535, scope: !0)
