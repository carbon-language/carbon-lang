; RUN: true
; Companion for debug-info-version-a.ll.

!llvm.module.flags = !{ !0 }
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 42}
!1 = !{!"0x11\0012\00clang\000\00", !"I AM UNEXPECTED!"} ; [ DW_TAG_compile_unit ]
!2 = !{!"b.c", !""}
!3 = !{}
