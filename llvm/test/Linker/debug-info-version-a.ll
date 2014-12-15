; RUN: llvm-link %s %p/debug-info-version-b.ll -S -o - | FileCheck %s

; Test linking of incompatible debug info versions. The debug info
; from the other file should be dropped.

; CHECK-NOT: metadata !{metadata !"b.c", metadata !""}
; CHECK:  !"a.c", !""}
; CHECK-NOT: metadata !{metadata !"b.c", metadata !""}

!llvm.module.flags = !{ !0 }
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 2}
!1 = !{!"0x11\0012\00clang\001\00\000\00\000", !2, !3, !3, !3, null, null} ; [ DW_TAG_compile_unit ]
!2 = !{!"a.c", !""}
!3 = !{}
