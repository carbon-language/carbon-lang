// Check miscellaneous Objective-C options.

// RUN: %clang -ccc-host-triple x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch x86_64 -fobjc-abi-version=1 2> %t
// RUN: FileCheck --check-prefix CHECK-X86_64_ABI1 < %t %s

// CHECK-CHECK-X86_64_ABI1: "-cc1"
// CHECK-CHECK-X86_64_ABI1: -fobjc-fragile-abi
// CHECK-CHECK-X86_64_ABI1-NOT: -fobjc-dispatch-method
// CHECK-CHECK-X86_64_ABI1: darwin-objc-options

// RUN: %clang -ccc-host-triple x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch i386 -fobjc-abi-version=2 2> %t
// RUN: FileCheck --check-prefix CHECK-I386_ABI2 < %t %s

// CHECK-CHECK-I386_ABI2: "-cc1"
// CHECK-CHECK-I386_ABI2-NOT: -fobjc-fragile-abi
// CHECK-CHECK-I386_ABI2: -fobjc-exceptions
// CHECK-CHECK-I386_ABI2: -fexceptions
// CHECK-CHECK-I386_ABI2-NOT: -fobjc-dispatch-method
// CHECK-CHECK-I386_ABI2: darwin-objc-options
