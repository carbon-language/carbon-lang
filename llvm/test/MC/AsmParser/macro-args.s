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


.macro top
    middle _$0, $1
.endm
.macro middle
     $0:
    .if $n > 1
        bottom $1
    .endif
.endm
.macro bottom
    .set fred, $0
.endm

.text

top foo
top bar, 42

// CHECK: _foo:
// CHECK-NOT: fred
// CHECK: _bar
// CHECK-NEXT: fred = 42
