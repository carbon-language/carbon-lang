// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t 2>&1 | FileCheck %s

// CHECK: error: Cannot represent a subtraction with a weak symbol

.weak f
.weak g
f:
    nop
g:
    nop

.quad g - f
