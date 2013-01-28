// RUN: %clang -### -S %s        2>&1 | FileCheck --check-prefix=CHECK-WITHOUT-G %s
// RUN: %clang -### -S %s -g     2>&1 | FileCheck --check-prefix=CHECK-WITH-G    %s
// RUN: %clang -### -S %s -g0    2>&1 | FileCheck --check-prefix=CHECK-WITHOUT-G %s
// RUN: %clang -### -S %s -g -g0 2>&1 | FileCheck --check-prefix=CHECK-WITHOUT-G %s
// RUN: %clang -### -S %s -g0 -g 2>&1 | FileCheck --check-prefix=CHECK-WITH-G    %s

// CHECK-WITHOUT-G-NOT: "-g"
// CHECK-WITH-G: "-g"

