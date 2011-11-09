// General tests that the header search paths detected by the driver and passed
// to CC1 are sane.
//
// Test a very broken version of multiarch that shipped in Ubuntu 11.04.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -ccc-host-triple i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/ubuntu_11.04_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-11-04 %s
// CHECK-UBUNTU-11-04: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-UBUNTU-11-04: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5/i686-linux-gnu"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5/backward"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-UBUNTU-11-04: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9]\.[0-9]}}/include"
// CHECK-UBUNTU-11-04: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-UBUNTU-11-04: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
