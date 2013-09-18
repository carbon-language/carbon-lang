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
