; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4, !4}
!named = !{!0, !1, !2, !3, !4, !5}

!0 = distinct !{}
!1 = !{!"path/to/file", !"/path/to/dir"}
!2 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")

; CHECK: !3 = !MDLexicalBlock(scope: !0, file: !2, line: 7, column: 35)
!3 = !MDLexicalBlock(scope: !0, file: !2, line: 7, column: 35)

; CHECK: !4 = !MDLexicalBlock(scope: !0)
!4 = !MDLexicalBlock(scope: !0)
!5 = !MDLexicalBlock(scope: !0, file: null, line: 0, column: 0)
