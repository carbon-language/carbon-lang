; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !7, !8, !8}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10}

!0 = distinct !{}
!1 = !MDFile(filename: "path/to/file", directory: "/path/to/dir")
!2 = distinct !{}
!3 = distinct !{}
!4 = distinct !{}
!5 = distinct !{}
!6 = distinct !{}

; CHECK: !7 = !MDCompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, flags: "-O2", runtimeVersion: 2, splitDebugFilename: "abc.debug", emissionKind: 3, enums: !2, retainedTypes: !3, subprograms: !4, globals: !5, imports: !6)
!7 = !MDCompileUnit(language: DW_LANG_C99, file: !1, producer: "clang",
                    isOptimized: true, flags: "-O2", runtimeVersion: 2,
                    splitDebugFilename: "abc.debug", emissionKind: 3,
                    enums: !2, retainedTypes: !3, subprograms: !4,
                    globals: !5, imports: !6)
!8 = !MDCompileUnit(language: 12, file: !1, producer: "clang",
                    isOptimized: true, flags: "-O2", runtimeVersion: 2,
                    splitDebugFilename: "abc.debug", emissionKind: 3,
                    enums: !2, retainedTypes: !3, subprograms: !4,
                    globals: !5, imports: !6)

; CHECK: !8 = !MDCompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: 0)
!9 = !MDCompileUnit(language: 12, file: !1, producer: "",
                    isOptimized: false, flags: "", runtimeVersion: 0,
                    splitDebugFilename: "", emissionKind: 0)
!10 = !MDCompileUnit(language: 12, file: !1)
