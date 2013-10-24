// RUN: %clang -target x86_64-apple-darwin -arch armv6m -dM -E %s | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -arch armv7m -dM -E %s | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -dM -E %s | FileCheck %s

// CHECK-NOT: __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__
// CHECK-NOT: __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
