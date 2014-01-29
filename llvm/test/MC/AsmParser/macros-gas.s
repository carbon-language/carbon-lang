// RUN: not llvm-mc -triple i386-linux-gnu %s 2> %t.err | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERRORS %s < %t.err

.macro .test0
.macrobody0
.endm
.macro .test1
.test0
.endm

.test1
// CHECK-ERRORS: <instantiation>:1:1: error: unknown directive
// CHECK-ERRORS-NEXT: macrobody0
// CHECK-ERRORS-NEXT: ^
// CHECK-ERRORS: <instantiation>:1:1: note: while in macro instantiation
// CHECK-ERRORS-NEXT: .test0
// CHECK-ERRORS-NEXT: ^
// CHECK-ERRORS: 11:1: note: while in macro instantiation
// CHECK-ERRORS-NEXT: .test1
// CHECK-ERRORS-NEXT: ^

.macro test2 _a
.byte \_a
.endm
// CHECK: .byte 10
test2 10

.macro test3 _a _b _c
.ascii "\_a \_b \_c \\_c"
.endm

// CHECK: .ascii "1 2 3 \003"
test3 1, 2, 3

// FIXME: test3 1, 2 3 should be treated like test 1, 2, 3

// FIXME: remove the n argument from the remaining test3 examples
// CHECK: .ascii "1 (23) n \n"
test3 1, (2 3), n

// CHECK: .ascii "1 (23) n \n"
test3 1 (2 3) n

// CHECK: .ascii "1 2 n \n"
test3 1 2 n

.macro test5 _a
.globl \_a
.endm

// CHECK: .globl zed1
test5 zed1

.macro test6 $a
.globl \$a
.endm

// CHECK: .globl zed2
test6 zed2

.macro test7 .a
.globl \.a
.endm

// CHECK: .globl zed3
test7 zed3

.macro test8 _a, _b, _c
.ascii "\_a,\_b,\_c"
.endm

.macro test9 _a _b _c
.ascii "\_a \_b \_c"
.endm

// CHECK: .ascii "a,b,c"
test8 a, b, c
// CHECK: .ascii "%1,%2,%3"
test8 %1 %2 %3 #a comment
// CHECK: .ascii "x-y,z,1"
test8 x - y z 1
// CHECK: .ascii "1 2 3"
test9 1, 2,3

test8 1,2 3
// CHECK-ERRORS: error: macro argument '_c' is missing
// CHECK-ERRORS-NEXT: test8 1,2 3
// CHECK-ERRORS-NEXT:           ^

test8 1 2, 3
// CHECK-ERRORS: error: expected ' ' for macro argument separator
// CHECK-ERRORS-NEXT:test8 1 2, 3
// CHECK-ERRORS-NEXT:         ^
