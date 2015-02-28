; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

@foo = global i32 0

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!0 = distinct !{}
!1 = distinct !{}
!2 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = distinct !{}
!4 = distinct !{}

; CHECK: !5 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "foo", arg: 3, scope: !0, file: !2, line: 7, type: !3, flags: DIFlagArtificial, inlinedAt: !4)
; CHECK: !6 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "foo", scope: !0, file: !2, line: 7, type: !3, flags: DIFlagArtificial, inlinedAt: !4)
!5 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "foo", arg: 3,
                      scope: !0, file: !2, line: 7, type: !3,
                      flags: DIFlagArtificial, inlinedAt: !4)
!6 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "foo", scope: !0,
                      file: !2, line: 7, type: !3, flags: DIFlagArtificial, inlinedAt: !4)

; CHECK: !7 = !MDLocalVariable(tag: DW_TAG_arg_variable, arg: 0, scope: null)
; CHECK: !8 = !MDLocalVariable(tag: DW_TAG_auto_variable, scope: null)
!7 = !MDLocalVariable(tag: DW_TAG_arg_variable)
!8 = !MDLocalVariable(tag: DW_TAG_auto_variable)
