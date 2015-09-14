// RUN: %clang %s --target=sparc-sun-solaris2.11 -### -o %t.o 2>&1 | FileCheck %s

// CHECK: "-fno-use-cxa-atexit"

