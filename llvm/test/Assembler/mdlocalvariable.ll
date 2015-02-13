; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

@foo = global i32 0

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!0 = distinct !{}
!1 = !{!"path/to/file", !"/path/to/dir"}
!2 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = distinct !{}
!4 = distinct !{}

; CHECK: !5 = !MDLocalVariable(tag: DW_TAG_arg_variable, scope: !0, name: "foo", file: !1, line: 7, type: !3, arg: 3, flags: 8, inlinedAt: !4)
; CHECK: !6 = !MDLocalVariable(tag: DW_TAG_auto_variable, scope: !0, name: "foo", file: !1, line: 7, type: !3, flags: 8, inlinedAt: !4)
!5 = !MDLocalVariable(tag: DW_TAG_arg_variable, scope: !0, name: "foo",
                      file: !1, line: 7, type: !3, arg: 3,
                      flags: 8, inlinedAt: !4)
!6 = !MDLocalVariable(tag: DW_TAG_auto_variable, scope: !0, name: "foo",
                      file: !1, line: 7, type: !3, flags: 8, inlinedAt: !4)

; CHECK: !7 = !MDLocalVariable(tag: DW_TAG_arg_variable, scope: null, name: "", arg: 0)
; CHECK: !8 = !MDLocalVariable(tag: DW_TAG_auto_variable, scope: null, name: "")
!7 = !MDLocalVariable(tag: DW_TAG_arg_variable)
!8 = !MDLocalVariable(tag: DW_TAG_auto_variable)
