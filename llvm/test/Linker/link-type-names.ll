; RUN: echo "%X = type { i32 } @G2 = global %X { i32 4 }" > %t.ll
; RUN: llvm-link %s %t.ll -S | FileCheck %s
; XFAIL: *
; PR11464

; FIXME: XFAIL until <rdar://problem/10913281> is addressed.

%X = type { i32 }
@G = global %X { i32 4 }


; CHECK: @G = global %X { i32 4 }
; CHECK: @G2 = global %X { i32 4 }
