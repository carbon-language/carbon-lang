; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

@foo = global i32 0

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6}
!named = !{!0, !1, !2, !3, !4, !5, !6}

!0 = distinct !{}
!1 = distinct !{}
!2 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = distinct !{}
!4 = distinct !{}

; CHECK: !5 = !MDGlobalVariable(name: "foo", linkageName: "foo", scope: !0, file: !2, line: 7, type: !3, isLocal: true, isDefinition: false, variable: i32* @foo, declaration: !4)
!5 = !MDGlobalVariable(name: "foo", linkageName: "foo", scope: !0,
                       file: !2, line: 7, type: !3, isLocal: true,
                       isDefinition: false, variable: i32* @foo,
                       declaration: !4)

; CHECK: !6 = !MDGlobalVariable(scope: null, isLocal: false, isDefinition: true)
!6 = !MDGlobalVariable()
