// RUN: %clang %s -target=x86_64-unknown-freebsd -### 2>&1 | FileCheck %s
// CHECK: ld
// RUN: %clang -fuse-ld=bfd --sysroot=%S/Inputs/basic_freebsd_tree \
// RUN:   -target=x86_64-unknown-freebsd \
// RUN:   -B%S/Inputs/basic_freebsd_tree %s  -### 2>&1 | \
// RUN:    FileCheck -check-prefix=CHECK-BFD %s
// CHECK-BFD: ld.bfd
// RUN: %clang -fuse-ld=gold --sysroot=%S/Inputs/basic_freebsd_tree \
// RUN:   -target=x86_64-unknown-freebsd \
// RUN:   -B%S/Inputs/basic_freebsd_tree %s -### 2>&1 | \
// RUN:    FileCheck -check-prefix=CHECK-GOLD %s
// CHECK-GOLD: ld.gold
// RUN: %clang -fuse-ld=plib --sysroot=%S/Inputs/basic_freebsd_tree \
// RUN:   -target=x86_64-unknown-freebsd \
// RUN:   -B%S/Inputs/basic_freebsd_tree %s -### 2>&1 | \
// RUN:    FileCheck -check-prefix=CHECK-PLIB %s
// CHECK-PLIB: error: invalid linker name
