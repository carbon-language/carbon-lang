// Tests that make sure armv7k is mapped to the correct CPU and ABI choices

// RUN: %clang -target x86_64-apple-macosx10.9 -arch armv7k -c %s -### 2>&1 | FileCheck %s
// CHECK: "-cc1"{{.*}} "-target-cpu" "cortex-a7"
// CHECK-NOT: "-exception-model=sjlj"

// "thumbv7k-apple-ios" is a bit of a weird triple, but since the backend is
// going to choose to use dwarf-based exceptions for it, the front-end needs to
// match.

// RUN: %clang -target x86_64-apple-macosx10.9 -arch armv7k -miphoneos-version-min=9.0 -c %s -### 2>&1 | FileCheck %s

// RUN: %clang -target x86_64-apple-macosx10.9 -arch armv7 -mwatchos-version-min=9.0 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SJLJ
// CHECK-SJLJ: "-exception-model=sjlj"
