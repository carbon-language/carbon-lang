// Check that we don't try to forward -Xclang or -mlinker-version to GCC.
//
// RUN: %clang -ccc-host-triple powerpc-unknown-unknown \
// RUN:   -ccc-clang-archs i386 -c %s \
// RUN:   -Xclang foo-bar \
// RUN:   -mlinker-version=10 -### 2> %t
// RUN: FileCheck < %t %s
//
// CHECK: gcc{{.*}}"
// CHECK-NOT: "-mlinker-version=10"
// CHECK-NOT: "-Xclang"
// CHECK-NOT: "foo-bar"
// CHECK: gcc_forward
