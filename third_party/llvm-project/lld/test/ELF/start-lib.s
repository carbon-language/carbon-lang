// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
// RUN:   %p/Inputs/start-lib1.s -o %t2.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
// RUN:   %p/Inputs/start-lib2.s -o %t3.o

// RUN: ld.lld -o %t3 %t1.o %t2.o %t3.o
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=TEST1 %s
// TEST1: Name: foo
// TEST1: Name: bar

// RUN: ld.lld -o %t3 %t1.o -u bar --start-lib %t2.o %t3.o
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=TEST2 %s
// TEST2-NOT: Name: foo
// TEST2: Name: bar

// RUN: ld.lld -o %t3 %t1.o --start-lib %t2.o %t3.o
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=TEST3 %s
// TEST3-NOT: Name: foo
// TEST3-NOT: Name: bar

// RUN: not ld.lld %t1.o --start-lib --start-lib 2>&1 | FileCheck -check-prefix=NESTED-LIB %s
// NESTED-LIB: nested --start-lib

// RUN: not ld.lld %t1.o --start-group --start-lib 2>&1 | FileCheck -check-prefix=LIB-IN-GROUP %s
// LIB-IN-GROUP: may not nest --start-lib in --start-group

// RUN: not ld.lld --end-lib 2>&1 | FileCheck -check-prefix=END %s
// END: stray --end-lib

.globl _start
_start:
