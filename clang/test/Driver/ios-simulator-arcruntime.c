// RUN: %clang -### -x objective-c -target i386-apple-darwin10 -arch i386 -mmacosx-version-min=10.6 -D__IPHONE_OS_VERSION_MIN_REQUIRED=40201 -fobjc-arc -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS1 %s
// RUN: %clang -### -x objective-c -target i386-apple-darwin10 -arch i386 -D__IPHONE_OS_VERSION_MIN_REQUIRED=50000 -fobjc-arc -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS2 %s
// 

// CHECK-OPTIONS1: i386-apple-macosx10.6.0
// CHECK-OPTIONS1: -fobjc-runtime=ios-4.2.1
// CHECK-OPTIONS2: i386-apple-macosx10.6.0
// CHECK-OPTIONS2: -fobjc-runtime=ios-5.0.0
