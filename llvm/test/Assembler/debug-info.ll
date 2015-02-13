; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !0, !1, !2, !3, !4, !5, !6, !7, !8, !8, !9, !10, !11, !12}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14}

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

; CHECK-NEXT: !6 = !MDBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
; CHECK-NEXT: !7 = !MDBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
; CHECK-NEXT: !8 = !MDBasicType(tag: DW_TAG_base_type)
!7 = !MDBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
!8 = !MDBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!9 = !MDBasicType(tag: DW_TAG_base_type)
!10 = !MDBasicType(tag: DW_TAG_base_type, name: "", size: 0, align: 0, encoding: 0)

; CHECK-NEXT: !9 = !{!"path/to/file", !"/path/to/dir"}
; CHECK-NEXT: !10 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
; CHECK-NEXT: !11 = !{null, null}
; CHECK-NEXT: !12 = !MDFile(filename: "", directory: "")
!11 = !{!"path/to/file", !"/path/to/dir"}
!12 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
!13 = !{null, null}
!14 = !MDFile(filename: "", directory: "")
