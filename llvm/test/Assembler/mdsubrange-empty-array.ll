; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !0, !1, !2}
!named = !{!0, !1, !2, !3}

; CHECK:      !0 = !MDSubrange(count: -1)
; CHECK-NEXT: !1 = !MDSubrange(count: -1, lowerBound: 4)
; CHECK-NEXT: !2 = !MDSubrange(count: -1, lowerBound: -5)
!0 = !MDSubrange(count: -1)
!1 = !MDSubrange(count: -1, lowerBound: 0)

!2 = !MDSubrange(count: -1, lowerBound: 4)
!3 = !MDSubrange(count: -1, lowerBound: -5)
