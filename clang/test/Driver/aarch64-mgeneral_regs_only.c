// Test the -mgeneral_regs_only option

// RUN: %clang -target aarch64-linux-eabi -mgeneral_regs_only %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FP %s
// CHECK-NO-FP: "-target-feature" "-fp-armv8"
// CHECK-NO-FP: "-target-feature" "-crypto"
// CHECK-NO-FP: "-target-feature" "-neon"
