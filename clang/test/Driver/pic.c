// Test the driver's control over the PIC behavior. These consist of tests of
// the relocation model flags and the pic level flags passed to CC1.
//
// CHECK-NO-PIC: "-mrelocation-model" "static"
// CHECK-NO-PIC-NOT: "-pic-level"
// CHECK-NO-PIC-NOT: "-pie-level"
//
// CHECK-DYNAMIC-NO-PIC1: "-mrelocation-model" "dynamic-no-pic"
// CHECK-DYNAMIC-NO-PIC1: "-pic-level" "1"
//
// CHECK-DYNAMIC-NO-PIC2: "-mrelocation-model" "dynamic-no-pic"
// CHECK-DYNAMIC-NO-PIC2: "-pic-level" "2"
//
// CHECK-PIC1-NOT: "-mrelocation-model"
// CHECK-PIC1: "-pic-level" "1"
//
// CHECK-PIC2-NOT: "-mrelocation-model"
// CHECK-PIC2: "-pic-level" "2"
//
// CHECK-PIE1-NOT: "-mrelocation-model"
// CHECK-PIE1: "-pie-level" "1"
//
// CHECK-PIE2-NOT: "-mrelocation-model"
// CHECK-PIE2: "-pie-level" "2"
//
// RUN: %clang -c %s -target i386-unknown-unknown -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fPIE -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fPIC -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// Defaults change for Darwin.
// RUN: %clang -c %s -target i386-apple-darwin -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-apple-darwin -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-apple-darwin -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
//
// Disregard any of the PIC-specific flags if we have a trump-card flag.
// RUN: %clang -c %s -target i386-unknown-unknown -mkernel -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -static -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -mdynamic-no-pic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC1
// RUN: %clang -c %s -target i386-apple-darwin -mdynamic-no-pic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC2
