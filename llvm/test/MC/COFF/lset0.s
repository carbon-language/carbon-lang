// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o - | llvm-nm - | FileCheck %s

not_global = 123
global = 456
.globl global
.Llocal = 789

// CHECK-NOT: not_global
// CHECK-NOT: Llocal
// CHECK: global
// CHECK-NOT: not_global
// CHECK-NOT: Llocal
