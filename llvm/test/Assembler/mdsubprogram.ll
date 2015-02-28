; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

declare void @_Z3foov()

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}

!0 = distinct !{}
!1 = distinct !{}
!2 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = distinct !{}
!4 = distinct !{}
!5 = distinct !{}
!6 = distinct !{}
!7 = distinct !{}

; CHECK: !8 = !MDSubprogram(name: "foo", linkageName: "_Zfoov", scope: !0, file: !2, line: 7, type: !3, isLocal: true, isDefinition: false, scopeLine: 8, containingType: !4, virtuality: DW_VIRTUALITY_pure_virtual, virtualIndex: 10, flags: DIFlagPrototyped, isOptimized: true, function: void ()* @_Z3foov, templateParams: !5, declaration: !6, variables: !7)
!8 = !MDSubprogram(name: "foo", linkageName: "_Zfoov", scope: !0,
                   file: !2, line: 7, type: !3, isLocal: true,
                   isDefinition: false, scopeLine: 8, containingType: !4,
                   virtuality: DW_VIRTUALITY_pure_virtual, virtualIndex: 10,
                   flags: DIFlagPrototyped, isOptimized: true, function: void ()* @_Z3foov,
                   templateParams: !5, declaration: !6, variables: !7)

; CHECK: !9 = !MDSubprogram(name: "bar", scope: null, isLocal: false, isDefinition: true, isOptimized: false)
!9 = !MDSubprogram(name: "bar")

