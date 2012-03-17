// RUN: llvm-mc -triple x86_64-apple-darwin10 %s | FileCheck %s

.macro GET   var,re2g
    movl   \var@GOTOFF(%ebx),\re2g
.endm


GET    is_sse, %eax

// CHECK: movl	is_sse@GOTOFF(%ebx), %eax

.macro bar
    .long $n
.endm

bar 1, 2, 3
bar

// CHECK: .long 3
// CHECK: .long 0
