// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %ta.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %tb.o
// RUN: llvm-mc -filetype=obj -triple=arm-unknown-linux %s -o %tc.o

// RUN: not lld -flavor gnu2 %ta.o %tb.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-B %s
// A-AND-B: a.o is incompatible with {{.*}}b.o

// RUN: not lld -flavor gnu2 %tb.o %tc.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=B-AND-C %s
// B-AND-C: b.o is incompatible with {{.*}}c.o

// FIMME: create the .so ourselves once we are able to
// RUN: not lld -flavor gnu2 %ta.o %p/Inputs/i686-simple-library.so -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-SO %s
// A-AND-SO: a.o is incompatible with {{.*}}/Inputs/i686-simple-library.so

// RUN: not lld -flavor gnu2 %tc.o %p/Inputs/i686-simple-library.so -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=C-AND-SO %s
// C-AND-SO: c.o is incompatible with {{.*}}/Inputs/i686-simple-library.so

// RUN: not lld -flavor gnu2 %p/Inputs/i686-simple-library.so %tc.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=SO-AND-C %s
// SO-AND-C: /Inputs/i686-simple-library.so is incompatible with {{.*}}c.o

// REQUIRES: x86,arm
