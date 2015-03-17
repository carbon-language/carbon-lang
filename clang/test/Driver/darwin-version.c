// RUN: %clang -target armv6-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX %s
// CHECK-VERSION-OSX: "armv6k-apple-macosx10.5.0"
// RUN: %clang -target armv6-apple-darwin9 -miphoneos-version-min=2.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS2 %s
// CHECK-VERSION-IOS2: "armv6k-apple-ios2.0.0"
// RUN: %clang -target armv6-apple-darwin9 -miphoneos-version-min=2.2 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS22 %s
// CHECK-VERSION-IOS22: "armv6k-apple-ios2.2.0"
// RUN: %clang -target armv6-apple-darwin9 -miphoneos-version-min=3.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS3 %s
// CHECK-VERSION-IOS3: "armv6k-apple-ios3.0.0"
// RUN: %clang -target i686-apple-darwin8 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX4 %s
// RUN: %clang -target i686-apple-darwin9 -mmacosx-version-min=10.4 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX4 %s
// CHECK-VERSION-OSX4: "i386-apple-macosx10.4.0"
// RUN: %clang -target i686-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
// RUN: %clang -target i686-apple-darwin9 -mmacosx-version-min=10.5 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
// CHECK-VERSION-OSX5: "i386-apple-macosx10.5.0"
// RUN: %clang -target i686-apple-darwin10 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX6 %s
// RUN: %clang -target i686-apple-darwin9 -mmacosx-version-min=10.6 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX6 %s
// CHECK-VERSION-OSX6: "i386-apple-macosx10.6.0"
// RUN: %clang -target x86_64-apple-darwin14 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
// RUN: %clang -target x86_64-apple-macosx -mmacosx-version-min=10.10 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
// CHECK-VERSION-OSX10: "x86_64-apple-macosx10.10.0"

// Check environment variable gets interpreted correctly
// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 \
// RUN:   %clang -target i386-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
// RUN: env IPHONEOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target armv6-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS2 %s

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.4.10 \
// RUN:   %clang -target i386-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX49 %s
// CHECK-VERSION-OSX49: "i386-apple-macosx10.4.10"
// RUN: env IPHONEOS_DEPLOYMENT_TARGET=2.3.1 \
// RUN:   %clang -target armv6-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS231 %s
// CHECK-VERSION-IOS231: "armv6k-apple-ios2.3.1"
