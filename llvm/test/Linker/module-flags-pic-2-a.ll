; RUN: not llvm-link %s %p/Inputs/module-flags-pic-2-b.ll -S -o - 2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

; test linking modules with two different PIC levels

!0 = !{ i32 1, !"PIC Level", i32 1 }

!llvm.module.flags = !{!0}

; CHECK-ERRORS: ERROR: linking module flags 'PIC Level': IDs have conflicting values
