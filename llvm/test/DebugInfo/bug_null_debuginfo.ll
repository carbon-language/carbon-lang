; RUN: llc < %s

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = !{!"0x11\0012\00\000\00\000\00\000", !1, null, null, null,  null, null} ; [ DW_TAG_compile_unit ]
!1 = !{!"t", !""}
!2 = !{i32 1, !"Debug Info Version", i32 2}
