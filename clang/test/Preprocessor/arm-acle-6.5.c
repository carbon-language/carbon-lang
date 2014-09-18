// RUN: %clang -target arm-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-DEFAULT

// CHECK-DEFAULT-NOT: __ARM_FP

// RUN: %clang -target arm-eabi -mfpu=vfp -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target arm-eabi -mfpu=vfp3 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target arm-eabi -mfpu=vfp3-d16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target arm-eabi -mfpu=neon -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target arm-eabi -mfpu=vfp3 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP
// RUN: %clang -target armv7-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP

// CHECK-SP-DP: __ARM_FP 0xC

// RUN: %clang -target arm-eabi -mfpu=vfpv4 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=vfpv4-d16 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=fp-armv8 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=neon-fp-armv8 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target arm-eabi -mfpu=crypto-neon-fp-armv8 -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP
// RUN: %clang -target armv8-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-SP-DP-HP

// CHECK-SP-DP-HP: __ARM_FP 0xE

