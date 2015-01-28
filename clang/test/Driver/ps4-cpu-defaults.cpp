// Check that on the PS4 we default to:
// -target-cpu btver2, -momit-leaf-frame-pointer, and no exceptions

// RUN: %clang -target x86_64-scei-ps4 -c %s -### 2>&1 | FileCheck %s
// CHECK: "-target-cpu" "btver2"
// CHECK: -momit-leaf-frame-pointer
// CHECK-NOT: exceptions
