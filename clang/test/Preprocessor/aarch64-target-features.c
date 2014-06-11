// RUN: %clang -target aarch64-none-linux-gnu -x c -E -dM %s -o - | FileCheck %s
// RUN: %clang -target arm64-none-linux-gnu -x c -E -dM %s -o - | FileCheck %s

// CHECK: __AARCH64EL__ 1
// CHECK: __ARM_64BIT_STATE 1
// CHECK: __ARM_ACLE 200
// CHECK: __ARM_ALIGN_MAX_STACK_PWR 4
// CHECK: __ARM_ARCH 8
// CHECK: __ARM_ARCH_ISA_A64 1
// CHECK: __ARM_ARCH_PROFILE 'A'
// CHECK-NOT: __ARM_FEATURE_BIG_ENDIAN
// CHECK: __ARM_FEATURE_CLZ 1
// CHECK-NOT: __ARM_FEATURE_CRC32 1
// CHECK-NOT: __ARM_FEATURE_CRYPTO 1
// CHECK: __ARM_FEATURE_DIV 1
// CHECK: __ARM_FEATURE_FMA 1
// CHECK: __ARM_FEATURE_UNALIGNED 1
// CHECK: __ARM_FP 0xe
// CHECK: __ARM_FP16_FORMAT_IEEE 1
// CHECK-NOT: __ARM_FP_FAST 1
// CHECK: __ARM_FP_FENV_ROUNDING 1
// CHECK: __ARM_NEON 1
// CHECK: __ARM_NEON_FP 0xe
// CHECK: __ARM_PCS_AAPCS64 1
// CHECK-NOT: __ARM_SIZEOF_MINIMAL_ENUM 1
// CHECK-NOT: __ARM_SIZEOF_WCHAR_T 2

// RUN: %clang -target aarch64-none-linux-gnu -mfpu=crypto-neon-fp-armv8 -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-CRYPTO %s
// RUN: %clang -target arm64-none-linux-gnu -mfpu=crypto-neon-fp-armv8 -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-CRYPTO %s
// CHECK-CRYPTO: __ARM_FEATURE_CRYPTO 1

// RUN: %clang -target aarch64-none-linux-gnu -mcrc -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-CRC32 %s
// RUN: %clang -target arm64-none-linux-gnu -mcrc -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-CRC32 %s
// CHECK-CRC32: __ARM_FEATURE_CRC32 1

// RUN: %clang -target aarch64-none-linux-gnu -ffast-math -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-FASTMATH %s
// RUN: %clang -target arm64-none-linux-gnu -ffast-math -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-FASTMATH %s
// CHECK-FASTMATH: __ARM_FP_FAST 1

// RUN: %clang -target aarch64-none-linux-gnu -fshort-wchar -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTWCHAR %s
// RUN: %clang -target arm64-none-linux-gnu -fshort-wchar -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTWCHAR %s
// CHECK-SHORTWCHAR: __ARM_SIZEOF_WCHAR_T 2

// RUN: %clang -target aarch64-none-linux-gnu -fshort-enums -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTENUMS %s
// RUN: %clang -target arm64-none-linux-gnu -fshort-enums -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTENUMS %s
// CHECK-SHORTENUMS: __ARM_SIZEOF_MINIMAL_ENUM 1

// RUN: %clang -target aarch64-none-linux-gnu -mfpu=neon -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-NEON %s
// RUN: %clang -target arm64-none-linux-gnu -mfpu=neon -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-NEON %s
// CHECK-NEON: __ARM_NEON 1
// CHECK-NEON: __ARM_NEON_FP 0xe

// RUN: %clang -target aarch64-none-linux-gnu -mcpu=cortex-a53 -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-FEATURE %s
// RUN: %clang -target aarch64-none-linux-gnu -mcpu=cortex-a57 -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-FEATURE %s
// RUN: %clang -target aarch64-none-linux-gnu -mcpu=cyclone -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-FEATURE %s
// CHECK-FEATURE: __ARM_FEATURE_CRC32 1
// CHECK-FEATURE: __ARM_FEATURE_CRYPTO 1
// CHECK-FEATURE: __ARM_NEON 1
// CHECK-FEATURE: __ARM_NEON_FP 0xe


