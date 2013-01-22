// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s 2> %t.err > %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s

.long $1
// CHECK-ERROR: 4:7: error: invalid token in expression
