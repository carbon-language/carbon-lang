// RUN: not %clang %s -target armv7-apple-ios -mfloat-abi=hard 2>&1 | FileCheck -check-prefix=ARMV7-ERROR %s
// RUN: %clang %s -target armv7-apple-ios -mfloat-abi=softfp -### 2>&1 | FileCheck -check-prefix=NOERROR %s
// RUN: %clang %s -arch armv7 -target thumbv7-apple-darwin-eabi -mfloat-abi=hard -### 2>&1 | FileCheck -check-prefix=NOERROR %s

// ARMV7-ERROR: unsupported option '-mfloat-abi=hard' for target 'thumbv7'
// NOERROR-NOT: unsupported option
