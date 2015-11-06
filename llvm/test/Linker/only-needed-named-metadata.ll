; RUN: llvm-as %S/only-needed-named-metadata.ll -o %t.bc
; RUN: llvm-as %S/Inputs/only-needed-named-metadata.ll -o %t2.bc
; RUN: llvm-link -S -only-needed %t2.bc %t.bc | FileCheck %s
; RUN: llvm-link -S -internalize -only-needed %t2.bc %t.bc | FileCheck %s

; CHECK: @U = external global i32
; CHECK: declare i32 @unused()

@X = global i32 5
@U = global i32 6
define i32 @foo() { ret i32 7 }
define i32 @unused() { ret i32 8 }

!llvm.named = !{!0, !1}
!0 = !{i32 ()* @unused}
!1 = !{i32* @U}
