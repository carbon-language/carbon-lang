; RUN: llvm-link %s %p/module-flags-8-b.ll -S -o - | sort | FileCheck %s

; Test append-type module flags.

; CHECK: !0 = metadata !{i32 5, metadata !"flag-0", metadata !1}
; CHECK: !1 = metadata !{i32 0, i32 0, i32 1}
; CHECK: !2 = metadata !{i32 6, metadata !"flag-1", metadata !3}
; CHECK: !3 = metadata !{i32 0, i32 1, i32 2}
; CHECK: !llvm.module.flags = !{!0, !2}

!0 = metadata !{ i32 5, metadata !"flag-0", metadata !{ i32 0 } }
!1 = metadata !{ i32 6, metadata !"flag-1", metadata !{ i32 0, i32 1 } }

!llvm.module.flags = !{ !0, !1 }
