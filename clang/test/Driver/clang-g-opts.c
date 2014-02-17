// RUN: %clang -### -S %s        2>&1 | FileCheck --check-prefix=CHECK-WITHOUT-G %s
// RUN: %clang -### -S %s -g -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G %s
// RUN: %clang -### -S %s -g -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G-DARWIN %s
// RUN: %clang -### -S %s -g0    2>&1 | FileCheck --check-prefix=CHECK-WITHOUT-G %s
// RUN: %clang -### -S %s -g -g0 2>&1 | FileCheck --check-prefix=CHECK-WITHOUT-G %s
// RUN: %clang -### -S %s -g0 -g -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G %s
// RUN: %clang -### -S %s -g0 -g -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G-DARWIN %s

// CHECK-WITHOUT-G-NOT: "-g"
// CHECK-WITH-G: "-g"
// CHECK-WITH-G-DARWIN: "-gdwarf-2"

