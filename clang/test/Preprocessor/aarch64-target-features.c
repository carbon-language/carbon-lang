// RUN: %clang -target aarch64-none-linux-gnu -x c -E -dM %s -o - | FileCheck %s
// CHECK: __AARCH64EL__
// CHECK-NOT: __AARCH_ADVSIMD_FP
// CHECK-NOT: __AARCH_FEATURE_ADVSIMD
// CHECK: __ARM_ACLE 101
// CHECK: __ARM_ARCH 8
// CHECK: __ARM_ARCH_PROFILE 'A'
// CHECK-NOT: __ARM_FEATURE_BIG_ENDIAN
// CHECK: __ARM_FEATURE_CLZ 1
// CHECK: __ARM_FEATURE_FMA 1
// CHECK: __ARM_FEATURE_LDREX 0xf
// CHECK: __ARM_FEATURE_UNALIGNED 1
// CHECK: __ARM_FP 0xe
// CHECK-NOT: __ARM_FP_FAST
// CHECK: __ARM_FP16_FORMAT_IEEE 1
// CHECK: __ARM_FP_FENV_ROUNDING 1
// CHECK-NOT: __ARM_NEON_FP
// CHECK-NOT: __ARM_NEON
// CHECK: __ARM_SIZEOF_MINIMAL_ENUM 4
// CHECK: __ARM_SIZEOF_WCHAR_T 4
// CHECK: __aarch64__


// RUN: %clang -target aarch64-none-linux-gnu -ffast-math -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-FASTMATH %s
// CHECK-FASTMATH: __ARM_FP_FAST

// RUN: %clang -target aarch64-none-linux-gnu -fshort-wchar -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTWCHAR %s
// CHECK-SHORTWCHAR: __ARM_SIZEOF_WCHAR_T 2

// RUN: %clang -target aarch64-none-linux-gnu -fshort-enums -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTENUMS %s
// CHECK-SHORTENUMS: __ARM_SIZEOF_MINIMAL_ENUM 1

