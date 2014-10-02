; RUN: llvm-link %s %p/debug-info-version-b.ll -S -o - | FileCheck %s

; Test linking of incompatible debug info versions. The debug info
; from the other file should be dropped.

; CHECK-NOT: metadata !{metadata !"b.c", metadata !""}
; CHECK: metadata !{metadata !"a.c", metadata !""}
; CHECK-NOT: metadata !{metadata !"b.c", metadata !""}

!llvm.module.flags = !{ !0 }
!llvm.dbg.cu = !{!1}

!0 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!1 = metadata !{i32 589841, metadata !2, i32 12, metadata !"clang", i1 true, metadata !"", i32 0, metadata !3, metadata !3, metadata !3, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!2 = metadata !{metadata !"a.c", metadata !""}
!3 = metadata !{}
