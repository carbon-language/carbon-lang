// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Check that we don't pass -mconstructor-aliases to CUDA device-side
// compilation, but we do pass it to host-side compilation.

// RUN: %clang -### -target x86_64-linux-gnu %s 2>&1 | FileCheck %s
// CHECK: "-cc1"
// CHECK-NOT: "-fcuda-is-device" {{.*}}"-mconstructor-aliases"
// CHECK-NOT: "-mconstructor-aliases" {{.*}}"-fcuda-is-device"
// CHECK: "-cc1"
// CHECK-SAME: "-mconstructor-aliases"
