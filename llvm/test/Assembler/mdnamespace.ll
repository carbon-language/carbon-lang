; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4, !4}
!named = !{!0, !1, !2, !3, !4, !5}

!0 = distinct !{}
!1 = distinct !{}
!2 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")

; CHECK: !3 = !MDNamespace(name: "Namespace", scope: !0, file: !2, line: 7)
!3 = !MDNamespace(name: "Namespace", scope: !0, file: !2, line: 7)

; CHECK: !4 = !MDNamespace(scope: !0)
!4 = !MDNamespace(name: "", scope: !0, file: null, line: 0)
!5 = !MDNamespace(scope: !0)
