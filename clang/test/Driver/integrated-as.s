// RUN: %clang -### -c -integrated-as %s 2>&1 | FileCheck %s

// REQUIRES: clang-driver

// CHECK: cc1as
// CHECK-NOT: -relax-all

// RUN: %clang -c -integrated-as -Wa,--compress-debug-sections %s 2>&1 | FileCheck --check-prefix=INVALID %s
// INVALID: error: unsupported argument '--compress-debug-sections' to option 'Wa,'
