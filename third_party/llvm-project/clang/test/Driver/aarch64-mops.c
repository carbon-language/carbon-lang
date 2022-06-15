// Test that target feature mops is implemented and available correctly
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.7-a+mops   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a        %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a+mops   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a+nomops %s 2>&1 | FileCheck %s --check-prefix=NO_MOPS
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.2-a+mops   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.3-a        %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.3-a+mops   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.3-a+nomops %s 2>&1 | FileCheck %s --check-prefix=NO_MOPS

// CHECK: "-target-feature" "+mops"
// NO_MOPS: "-target-feature" "-mops"
