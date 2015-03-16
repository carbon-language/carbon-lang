; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4, !4}
!named = !{!0, !1, !2, !3, !4, !5}

!0 = distinct !{}
!1 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
!2 = distinct !{}


; CHECK: !2 = distinct !{}
; CHECK-NEXT: !3 = !MDObjCProperty(name: "foo", file: !1, line: 7, setter: "setFoo", getter: "getFoo", attributes: 7, type: !2)
!3 = !MDObjCProperty(name: "foo", file: !1, line: 7, setter: "setFoo",
                     getter: "getFoo", attributes: 7, type: !2)

; CHECK-NEXT: !4 = !MDObjCProperty()
!4 = !MDObjCProperty(name: "", file: null, line: 0, setter: "", getter: "",
                     attributes: 0, type: null)
!5 = !MDObjCProperty()
