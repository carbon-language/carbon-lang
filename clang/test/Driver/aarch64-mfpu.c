// Test that different values of -mfpu pick correct AArch64 FPU target-feature(s).

// RUN: %clang -target aarch64-linux-eabi -mfpu=neon %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON %s
// RUN: %clang -target aarch64-linux-eabi %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON %s
// CHECK-NEON: "-target-feature" "+neon"

// RUN: %clang -target aarch64-linux-eabi -mfpu=fp-armv8 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP-ARMV8 %s
// CHECK-FP-ARMV8: "-target-feature" "+fp-armv8"

// RUN: %clang -target aarch64-linux-eabi -mfpu=neon-fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON-FP-ARMV8 %s
// CHECK-NEON-FP-ARMV8: "-target-feature" "+fp-armv8"
// CHECK-NEON-FP-ARMV8: "-target-feature" "+neon"

// RUN: %clang -target aarch64-linux-eabi -mfpu=crypto-neon-fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRYPTO-NEON-FP-ARMV8 %s
// CHECK-CRYPTO-NEON-FP-ARMV8: "-target-feature" "+fp-armv8"
// CHECK-CRYPTO-NEON-FP-ARMV8: "-target-feature" "+neon"
// CHECK-CRYPTO-NEON-FP-ARMV8: "-target-feature" "+crypto"

