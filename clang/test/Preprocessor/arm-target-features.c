// RUN: %clang -target armv8a-none-linux-gnu -x c -E -dM %s -o - | FileCheck %s
// CHECK: __ARMEL__ 1
// CHECK: __ARM_ARCH 8
// CHECK: __ARM_ARCH_8A__ 1
// CHECK: __ARM_FEATURE_CRC32 1

// RUN: %clang -target armv7a-none-linux-gnu -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-V7 %s
// CHECK-V7: __ARMEL__ 1
// CHECK-V7: __ARM_ARCH 7
// CHECK-V7: __ARM_ARCH_7A__ 1
// CHECK-NOT-V7: __ARM_FEATURE_CRC32

// RUN: %clang -target armv8a -mfloat-abi=hard -x c -E -dM %s | FileCheck --check-prefix=CHECK-V8-BAREHF %s
// CHECK-V8-BAREHF: __ARMEL__ 1
// CHECK-V8-BAREHF: __ARM_ARCH 8
// CHECK-V8-BAREHF: __ARM_ARCH_8A__ 1
// CHECK-V8-BAREHF: __ARM_FEATURE_CRC32 1
// CHECK-V8-BAREHF: __ARM_NEON__ 1
// CHECK-V8-BAREHF: __VFP_FP__ 1

// RUN: %clang -target armv8a -mfloat-abi=hard -mfpu=fp-armv8 -x c -E -dM %s | FileCheck --check-prefix=CHECK-V8-BAREHF-FP %s
// CHECK-V8-BAREHF-FP-NOT: __ARM_NEON__ 1
// CHECK-V8-BAREHF-FP: __VFP_FP__ 1

// RUN: %clang -target armv8a -mfloat-abi=hard -mfpu=neon-fp-armv8 -x c -E -dM %s | FileCheck --check-prefix=CHECK-V8-BAREHF-NEON-FP %s
// RUN: %clang -target armv8a -mfloat-abi=hard -mfpu=crypto-neon-fp-armv8 -x c -E -dM %s | FileCheck --check-prefix=CHECK-V8-BAREHF-NEON-FP %s
// CHECK-V8-BAREHF-NEON-FP: __ARM_NEON__ 1
// CHECK-V8-BAREHF-NEON-FP: __VFP_FP__ 1

// RUN: %clang -target armv8a -mnocrc -x c -E -dM %s | FileCheck --check-prefix=CHECK-V8-NOCRC %s
// CHECK-V8-NOCRC-NOT: __ARM_FEATURE_CRC32 1

// Check that -mhwdiv works properly for armv8/thumbv8 (enabled by default).

// RUN: %clang -target armv8 -x c -E -dM %s -o - | FileCheck --check-prefix=ARMV8 %s
// ARMV8:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv8 -mthumb -x c -E -dM %s -o - | FileCheck --check-prefix=THUMBV8 %s
// THUMBV8:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv8-eabi -x c -E -dM %s -o - | FileCheck --check-prefix=ARMV8-EABI %s
// ARMV8-EABI:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv8-eabi -mthumb -x c -E -dM %s -o - | FileCheck --check-prefix=THUMBV8-EABI %s
// THUMBV8-EABI:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv8 -mhwdiv=none -x c -E -dM %s -o - | FileCheck --check-prefix=NONEHWDIV-ARMV8 %s
// NONEHWDIV-ARMV8-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target armv8 -mthumb -mhwdiv=none -x c -E -dM %s -o - | FileCheck --check-prefix=NONEHWDIV-THUMBV8 %s
// NONEHWDIV-THUMBV8-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target armv8 -mhwdiv=thumb -x c -E -dM %s -o - | FileCheck --check-prefix=THUMBHWDIV-ARMV8 %s
// THUMBHWDIV-ARMV8-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target armv8 -mthumb -mhwdiv=arm -x c -E -dM %s -o - | FileCheck --check-prefix=ARMHWDIV-THUMBV8 %s
// ARMHWDIV-THUMBV8-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target armv8a -x c -E -dM %s -o - | FileCheck --check-prefix=ARMV8A %s
// ARMV8A:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv8a -mthumb -x c -E -dM %s -o - | FileCheck --check-prefix=THUMBV8A %s
// THUMBV8A:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv8a-eabi -x c -E -dM %s -o - | FileCheck --check-prefix=ARMV8A-EABI %s
// ARMV8A-EABI:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv8a-eabi -x c -E -dM %s -o - | FileCheck --check-prefix=THUMBV8A-EABI %s
// THUMBV8A-EABI:#define __ARM_ARCH_EXT_IDIV__ 1


// Test that -mhwdiv has the right effect for a target CPU which has hwdiv enabled by default.
// RUN: %clang -target armv7 -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck --check-prefix=DEFAULTHWDIV-ARM %s
// DEFAULTHWDIV-ARM:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck --check-prefix=DEFAULTHWDIV-THUMB %s
// DEFAULTHWDIV-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv7 -mcpu=cortex-a15 -mhwdiv=arm -x c -E -dM %s -o - | FileCheck --check-prefix=ARMHWDIV-ARM %s
// ARMHWDIV-ARM:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a15 -mhwdiv=thumb -x c -E -dM %s -o - | FileCheck --check-prefix=THUMBHWDIV-THUMB %s
// THUMBHWDIV-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target arm -mcpu=cortex-a15 -mhwdiv=thumb -x c -E -dM %s -o - | FileCheck --check-prefix=DEFAULTHWDIV-THUMBHWDIV-ARM %s
// DEFAULTHWDIV-THUMBHWDIV-ARM-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target arm -mthumb -mcpu=cortex-a15 -mhwdiv=arm -x c -E -dM %s -o - | FileCheck --check-prefix=DEFAULTHWDIV-ARMHWDIV-THUMB %s
// DEFAULTHWDIV-ARMHWDIV-THUMB-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target arm -mcpu=cortex-a15 -mhwdiv=none -x c -E -dM %s -o - | FileCheck --check-prefix=DEFAULTHWDIV-NONEHWDIV-ARM %s
// DEFAULTHWDIV-NONEHWDIV-ARM-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target arm -mthumb -mcpu=cortex-a15 -mhwdiv=none -x c -E -dM %s -o - | FileCheck --check-prefix=DEFAULTHWDIV-NONEHWDIV-THUMB %s
// DEFAULTHWDIV-NONEHWDIV-THUMB-NOT:#define __ARM_ARCH_EXT_IDIV__

// FIXME: add check for further predefines
// Test whether predefines are as expected when targeting cortex-a5.
// RUN: %clang -target armv7 -mcpu=cortex-a5 -x c -E -dM %s -o - | FileCheck --check-prefix=A5-ARM %s
// A5-ARM-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a5 -x c -E -dM %s -o - | FileCheck --check-prefix=A5-THUMB %s
// A5-THUMB-NOT:#define __ARM_ARCH_EXT_IDIV__

// Test whether predefines are as expected when targeting cortex-a8.
// RUN: %clang -target armv7 -mcpu=cortex-a8 -x c -E -dM %s -o - | FileCheck --check-prefix=A8-ARM %s
// A8-ARM-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a8 -x c -E -dM %s -o - | FileCheck --check-prefix=A8-THUMB %s
// A8-THUMB-NOT:#define __ARM_ARCH_EXT_IDIV__

// Test whether predefines are as expected when targeting cortex-a9.
// RUN: %clang -target armv7 -mcpu=cortex-a9 -x c -E -dM %s -o - | FileCheck --check-prefix=A9-ARM %s
// A9-ARM-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a9 -x c -E -dM %s -o - | FileCheck --check-prefix=A9-THUMB %s
// A9-THUMB-NOT:#define __ARM_ARCH_EXT_IDIV__

// Test whether predefines are as expected when targeting cortex-a15.
// RUN: %clang -target armv7 -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck --check-prefix=A15-ARM %s
// A15-ARM:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck --check-prefix=A15-THUMB %s
// A15-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1

// Test whether predefines are as expected when targeting swift.
// RUN: %clang -target armv7s -mcpu=swift -x c -E -dM %s -o - | FileCheck --check-prefix=SWIFT-ARM %s
// SWIFT-ARM:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv7s -mthumb -mcpu=swift -x c -E -dM %s -o - | FileCheck --check-prefix=SWIFT-THUMB %s
// SWIFT-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1

// Test whether predefines are as expected when targeting cortex-a53.
// RUN: %clang -target armv8 -mcpu=cortex-a53 -x c -E -dM %s -o - | FileCheck --check-prefix=A53-ARM %s
// A53-ARM:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv8 -mthumb -mcpu=cortex-a53 -x c -E -dM %s -o - | FileCheck --check-prefix=A53-THUMB %s
// A53-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1

// Test whether predefines are as expected when targeting cortex-r5.
// RUN: %clang -target armv7 -mcpu=cortex-r5 -x c -E -dM %s -o - | FileCheck --check-prefix=R5-ARM %s
// R5-ARM:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv7 -mthumb -mcpu=cortex-r5 -x c -E -dM %s -o - | FileCheck --check-prefix=R5-THUMB %s
// R5-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1

// Test whether predefines are as expected when targeting cortex-m0.
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-m0 -x c -E -dM %s -o - | FileCheck --check-prefix=M0-THUMB %s
// M0-THUMB-NOT:#define __ARM_ARCH_EXT_IDIV__

// Test whether predefines are as expected when targeting cortex-m3.
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-m3 -x c -E -dM %s -o - | FileCheck --check-prefix=M3-THUMB %s
// M3-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1

// Test whether predefines are as expected when targeting cortex-m4.
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-m4 -x c -E -dM %s -o - | FileCheck --check-prefix=M4-THUMB %s
// M4-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1
