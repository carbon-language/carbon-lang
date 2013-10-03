// RUN: %clang -arch armv7 -target thumbv7-apple-darwin-eabi -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-AAPCS
// RUN: %clang -arch armv7s -target thumbv7-apple-ios -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-APCS
// RUN: %clang -arch armv7s -target thumbv7-apple-darwin -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-APCS

// CHECK-AAPCS: "-target-abi" "aapcs"
// CHECK-APCS: "-target-abi" "apcs-gnu"
