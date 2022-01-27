; RUN: llvm-link %s -S | FileCheck %s

; CHECK-DAG: @a = global i32 0
; CHECK-DAG: @b = global i32 0, !associated !0

; CHECK-DAG: !0 = !{i32* @b}

@a = global i32 0
@b = global i32 0, !associated !0

!0 = !{i32* @b}
