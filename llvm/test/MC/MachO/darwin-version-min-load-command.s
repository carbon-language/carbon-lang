// RUN: llvm-mc -triple x86_64-apple-macosx10.10.0 %s -filetype=obj -o - | llvm-objdump -macho -private-headers - | FileCheck %s
// RUN: llvm-mc -triple x86_64-apple-ios8.0.0 %s -filetype=obj -o - | llvm-objdump -macho -private-headers - | FileCheck %s --check-prefix=CHECK-IOS
// RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o - | llvm-objdump -macho -private-headers - | FileCheck %s --check-prefix=CHECK-DARWIN

// Test version-min load command should be inferred from triple and should always be generated on Darwin
// CHECK:           cmd LC_VERSION_MIN_MACOSX
// CHECK-NEXT:   cmdsize 16
// CHECK-NEXT:   version 10.10

// CHECK-IOS:           cmd LC_VERSION_MIN_IPHONEOS
// CHECK-IOS-NEXT:   cmdsize 16
// CHECK-IOS-NEXT:   version 8.0

// CHECK-DARWIN-NOT: LC_VERSION_MIN
