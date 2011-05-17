// RUN: %clang -ccc-host-triple i386-apple-darwin10  -static -### %s 2>&1 | \
// RUN:  FileCheck --check-prefix=CHECK-DARWIN %s

// RUN: %clang -ccc-host-triple i386-pc-linux-gnu  -static -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-LINUX %s

// CHECK-DARWIN: -fno-dwarf2-cfi-asm
// CHECK-LINUX-NOT: -fno-dwarf2-cfi-asm
