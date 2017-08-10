// REQUIRES: mips-registered-target
// RUN: %clang -### -c -target mips64-mti-elf -fno-PIC -mabicalls %s 2>&1 | FileCheck %s
// CHECK: warning: ignoring '-mabicalls' option as it cannot be used with non position-independent code and the N64 ABI
