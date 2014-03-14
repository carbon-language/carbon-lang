// RUN: %clang -x c-header %s -o %t.pch -MMD -MT dependencies -MF %t.d -### 2> %t
// RUN: FileCheck %s -input-file=%t
// CHECK: -emit-pch
// CHECK: -dependency-file
// CHECK: -module-file-deps

// RUN: %clang -c %s -o %t -MMD -MT dependencies -MF %t.d -### 2> %t
// RUN: FileCheck %s -check-prefix=CHECK-NOPCH -input-file=%t
// CHECK-NOPCH: -dependency-file
// CHECK-NOPCH-NOT: -module-file-deps
