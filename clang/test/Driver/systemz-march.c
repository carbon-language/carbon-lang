// Check that -march works for all supported targets.

// RUN: not %clang -target s390x -S -emit-llvm -march=z9 %s -o - 2>&1 | FileCheck --check-prefix=CHECK-Z9 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=z10 %s 2>&1 | FileCheck --check-prefix=CHECK-Z10 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=z196 %s 2>&1 | FileCheck --check-prefix=CHECK-Z196 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=zEC12 %s 2>&1 | FileCheck --check-prefix=CHECK-ZEC12 %s

// CHECK-Z9: error: unknown target CPU 'z9'
// CHECK-Z10: "-target-cpu" "z10"
// CHECK-Z196: "-target-cpu" "z196"
// CHECK-ZEC12: "-target-cpu" "zEC12"

int x;
