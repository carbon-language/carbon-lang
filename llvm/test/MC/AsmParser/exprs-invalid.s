// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s 2> %t.err | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERRORS %s < %t.err
// CHECK: 	.section	__TEXT,__text,regular,pure_instructions
// CHECK-ERRORS: error: Invalid octal number
.long 80+08
