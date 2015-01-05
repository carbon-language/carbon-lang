// RUN: %clang -v --target=i386-unknown-linux \
// RUN:           --gcc-toolchain="" \
// RUN:           --sysroot=%S/Inputs/debian_multiarch_tree 2>&1 | FileCheck %s

// CHECK: Found candidate GCC installation: {{.*}}Inputs{{.}}debian_multiarch_tree{{.}}usr{{.}}lib{{.}}gcc{{.}}i686-linux-gnu{{.}}4.5
// CHECK-NEXT: Found candidate GCC installation: {{.*}}Inputs{{.}}debian_multiarch_tree{{.}}usr{{.}}lib{{.}}gcc{{.}}x86_64-linux-gnu{{.}}4.5
// CHECK-NEXT: Selected GCC installation: {{.*}}Inputs{{.}}debian_multiarch_tree{{.}}usr{{.}}lib{{.}}gcc{{.}}i686-linux-gnu{{.}}4.5
