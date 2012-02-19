// Test that gcc-toolchain option is working correctly
//
// RUN: %clangxx -no-canonical-prefixes %s -### -o %t 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:     -gcc-toolchain %S/Inputs/ubuntu_11.04_multiarch_tree/usr \
// RUN:   | FileCheck %s

// CHECK: "-internal-isystem"
// CHECK: "[[TOOLCHAIN:.*]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5"
// CHECK: "-internal-isystem"
// CHECK: "[[TOOLCHAIN]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5/i686-linux-gnu"
// CHECK: "-internal-isystem"
// CHECK: "[[TOOLCHAIN]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5/backward"
// CHECK: "-internal-isystem"
// CHECK: "/usr/local/include"
// CHECK: "-internal-isystem"
// CHECK: lib/clang/3.1/include"
// CHECK: "-internal-externc-isystem"
// CHECK: "/include"
// CHECK: "-internal-externc-isystem"
// CHECK: "/usr/include"
// CHECK: "{{.*}}/ld"
// CHECK: "[[TOOLCHAIN]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/crtbegin.o"
// CHECK: "-L[[TOOLCHAIN]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5"
// CHECK: "-L[[TOOLCHAIN]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../.."
// CHECK: "-L/lib"
// CHECK: "-L/usr/lib"
