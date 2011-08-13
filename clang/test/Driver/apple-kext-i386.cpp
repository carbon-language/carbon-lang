// Check that we transparently fallback to llvm-gcc for i386 kexts, we don't
// support the ABI they use (yet).

// RUN: %clang -ccc-host-triple i386-apple-darwin10 \
// RUN:   -fapple-kext -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK < %t %s

// CHECK: cc1plus"
// CHECK: "-fapple-kext"

// RUN: %clang -ccc-host-triple i386-apple-darwin10 \
// RUN:   -mkernel -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MKERNEL < %t %s

// CHECK-MKERNEL: cc1plus"
// CHECK-MKERNEL: "-mkernel"
