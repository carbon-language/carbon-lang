; RUN: llvm-link %s %p/debug-info-version-b.ll -S -o - | FileCheck %s

; Test linking of incompatible debug info versions. The debug info
; from the other file should be dropped.

; CHECK-NOT: metadata !{metadata !"b.c", metadata !""}
; CHECK: metadata !{metadata !"a.c", metadata !""}
; CHECK-NOT: metadata !{metadata !"b.c", metadata !""}

!llvm.module.flags = !{ !0 }
!llvm.dbg.cu = !{!1}

!0 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!1 = metadata !{metadata !"0x11\0012\00clang\001\00\000\00\000", metadata !2, metadata !3, metadata !3, metadata !3, null, null} ; [ DW_TAG_compile_unit ]
!2 = metadata !{metadata !"a.c", metadata !""}
!3 = metadata !{}
