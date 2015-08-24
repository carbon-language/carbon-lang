// RUN: not %clang %s -target armv7-apple-ios -mfloat-abi=hard 2>&1 | FileCheck -check-prefix=ARMV7-HARD %s
// RUN: %clang %s -target armv7-apple-ios -mfloat-abi=softfp -### 2>&1 | FileCheck -check-prefix=ARMV7-SOFTFP %s

// ARMV7-HARD: unsupported option '-mfloat-abi=hard' for target 'thumbv7'
// ARMV7-SOFTFP-NOT: unsupported option
