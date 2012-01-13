// RUN: %clang -ccc-host-triple i386-apple-darwin10 -### -fsyntax-only -fgnu-runtime %s 2>&1 | FileCheck %s
// RUN: %clang -ccc-host-triple i386-apple-darwin10 -### -x objective-c++ -fsyntax-only -fgnu-runtime %s 2>&1 | FileCheck %s
// CHECK: -fgnu-runtime
// CHECK: -fobjc-runtime-has-arc
// CHECK: -fobjc-runtime-has-weak
