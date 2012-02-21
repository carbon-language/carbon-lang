// Test that gcc-toolchain option is working correctly
//
// RUN: %clangxx -no-canonical-prefixes %s -### -o %t 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:     -gcc-toolchain %S/Inputs/ubuntu_11.04_multiarch_tree/usr \
// RUN:   | FileCheck %s
//
// Test for header search toolchain detection.
// CHECK: "-internal-isystem"
// CHECK: "[[TOOLCHAIN:[^"]+]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5"
// CHECK: "-internal-isystem"
// CHECK: "[[TOOLCHAIN]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5/i686-linux-gnu"
// CHECK: "-internal-isystem"
// CHECK: "[[TOOLCHAIN]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5/backward"
// CHECK: "-internal-isystem" "/usr/local/include"
//
// Test for linker toolchain detection. Note that we use a separate variable
// because the '/'s may be different in the linker invocation than in the
// header search.
// CHECK: "{{[^"]*}}ld{{(.exe)?}}"
// CHECK: "[[TOOLCHAIN2:[^"]*]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/crtbegin.o"
// CHECK: "-L[[TOOLCHAIN2]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5"
// CHECK: "-L[[TOOLCHAIN2]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../.."
