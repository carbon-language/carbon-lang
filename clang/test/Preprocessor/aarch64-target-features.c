// RUN: %clang -target aarch64-none-linux-gnu -x c -E -dM %s -o - | FileCheck %s
// CHECK: __AARCH 8
// CHECK: __AARCH64EL__
// CHECK: __AARCH_ACLE 101
// CHECK-NOT: __AARCH_ADVSIMD_FP
// CHECK-NOT: __AARCH_FEATURE_ADVSIMD
// CHECK-NOT: __AARCH_FEATURE_BIG_ENDIAN
// CHECK: __AARCH_FEATURE_CLZ 1
// CHECK: __AARCH_FEATURE_FMA 1
// CHECK: __AARCH_FEATURE_LDREX 0xf
// CHECK: __AARCH_FEATURE_UNALIGNED 1
// CHECK: __AARCH_FP 0xe
// CHECK-NOT: __AARCH_FP_FAST
// CHECK: __AARCH_FP16_FORMAT_IEEE 1
// CHECK: __AARCH_FP_FENV_ROUNDING 1
// CHECK: __AARCH_PROFILE 'A'
// CHECK: __AARCH_SIZEOF_MINIMAL_ENUM 4
// CHECK: __AARCH_SIZEOF_WCHAR_T 4
// CHECK: __aarch64__


// RUN: %clang -target aarch64-none-linux-gnu -ffast-math -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-FASTMATH %s
// CHECK-FASTMATH: __AARCH_FP_FAST

// RUN: %clang -target aarch64-none-linux-gnu -fshort-wchar -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTWCHAR %s
// CHECK-SHORTWCHAR: __AARCH_SIZEOF_WCHAR_T 2

// RUN: %clang -target aarch64-none-linux-gnu -fshort-enums -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTENUMS %s
// CHECK-SHORTENUMS: __AARCH_SIZEOF_MINIMAL_ENUM 1

