; This file is used with module-flags-6-a.ll
; RUN: true

!0 = metadata !{ i32 5, metadata !"flag-0", metadata !{ i32 0, i32 1 } }
!1 = metadata !{ i32 6, metadata !"flag-1", metadata !{ i32 1, i32 2 } }

!llvm.module.flags = !{ !0, !1 }
