// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s 2> %t.err | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERRORS %s < %t.err

.macro .test0
.macrobody0
.endmacro
.macro .test1
.test0
.endmacro

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

.macro test2
.byte $0
.endmacro
test2 10

.macro test3
.globl "$0 $1 $2 $$3 $n"
.endmacro

// CHECK: .globl	"1 23  $3 2"
test3 1,2 3

.macro test4
.globl "$0 -- $1"
.endmacro

// CHECK: .globl	"ab)(,) -- (cd)"
test4 a b)(,),(cd)

.macro test5 _a
.globl "\_a"
.endm

test5 zed1
// CHECK: .globl zed1

.macro test6 $a
.globl "\$a"
.endm

test6 zed2
// CHECK: .globl zed2

.macro test7 .a
.globl "\.a"
.endm

test7 zed3
// CHECK: .globl zed3
