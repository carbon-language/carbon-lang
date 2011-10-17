// RUN: %clang -### -x objective-c -ccc-host-triple i386-apple-darwin10 -arch i386 -mios-simulator-version-min=4.2 -fobjc-arc -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS1 %s
// RUN: %clang -### -x objective-c -ccc-host-triple i386-apple-darwin10 -arch i386 -mios-simulator-version-min=5 -fobjc-arc -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS2 %s
// 

// CHECK-OPTIONS1-NOT: -fobjc-runtime-has-weak
// CHECK-OPTIONS2: -fobjc-runtime-has-weak
