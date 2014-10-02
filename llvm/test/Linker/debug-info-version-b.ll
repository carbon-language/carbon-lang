; RUN: true
; Companion for debug-info-version-a.ll.

!llvm.module.flags = !{ !0 }
!llvm.dbg.cu = !{!1}

!0 = metadata !{i32 2, metadata !"Debug Info Version", i32 42}
!1 = metadata !{i32 589841, metadata !2, i32 12, metadata !"clang", metadata !"I AM UNEXPECTED!"} ; [ DW_TAG_compile_unit ]
!2 = metadata !{metadata !"b.c", metadata !""}
!3 = metadata !{}
