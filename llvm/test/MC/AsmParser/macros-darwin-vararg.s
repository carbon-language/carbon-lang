// RUN: not llvm-mc -triple i386-apple-darwin10 %s 2>&1 | FileCheck %s

// CHECK: error: vararg is not a valid parameter qualifier for 'arg' in macro 'abc'
// CHECK: .macro abc arg:vararg

.macro abc arg:vararg
    \arg
.endm
