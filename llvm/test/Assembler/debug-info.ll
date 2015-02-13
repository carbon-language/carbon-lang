; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !0, !1, !2, !3, !4, !5}
!named = !{!0, !1, !2, !3, !4, !5, !6}

; CHECK:      !0 = !MDSubrange(count: 3)
; CHECK-NEXT: !1 = !MDSubrange(count: 3, lowerBound: 4)
; CHECK-NEXT: !2 = !MDSubrange(count: 3, lowerBound: -5)
!0 = !MDSubrange(count: 3)
!1 = !MDSubrange(count: 3, lowerBound: 0)

!2 = !MDSubrange(count: 3, lowerBound: 4)
!3 = !MDSubrange(count: 3, lowerBound: -5)

; CHECK-NEXT: !3 = !MDEnumerator(value: 7, name: "seven")
; CHECK-NEXT: !4 = !MDEnumerator(value: -8, name: "negeight")
; CHECK-NEXT: !5 = !MDEnumerator(value: 0, name: "")
!4 = !MDEnumerator(value: 7, name: "seven")
!5 = !MDEnumerator(value: -8, name: "negeight")
!6 = !MDEnumerator(value: 0, name: "")
