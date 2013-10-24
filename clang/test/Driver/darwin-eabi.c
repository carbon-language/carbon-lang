// RUN: %clang -arch armv7 -target thumbv7-apple-darwin-eabi -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-AAPCS
// RUN: %clang -arch armv7s -target thumbv7-apple-ios -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-APCS
// RUN: %clang -arch armv7s -target thumbv7-apple-darwin -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-APCS
// RUN: %clang -arch armv6m -target thumbv7-apple-darwin -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-DARWIN-EABI
// RUN: %clang -arch armv7m -target thumbv7-apple-darwin -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-DARWIN-EABI
// RUN: %clang -arch armv7em -target thumbv7-apple-darwin -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-DARWIN-EABI

// CHECK-DARWIN-EABI: "-triple" "{{thumbv[67]e?m}}-apple-darwin-eabi"
// CHECK-IOS: "-triple" "thumbv7" "thumbv7-apple-ios

// CHECK-AAPCS: "-target-abi" "aapcs"
// CHECK-APCS: "-target-abi" "apcs-gnu"
