; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK: invalid retained type
; CHECK: warning: ignoring invalid debug info

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.dbg.cu = !{!1}
!1 = distinct !DICompileUnit(file: !2, language: DW_LANG_C99, retainedTypes: !3)
!2 = !DIFile(filename: "file.c", directory: "/path/to/dir")
!3 = !{null}
