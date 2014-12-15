; This file is used with module-flags-4-a.ll
; RUN: true

!0 = !{i32 3, !"foo", !{!"bar", i32 42}}

!llvm.module.flags = !{ !0 }
