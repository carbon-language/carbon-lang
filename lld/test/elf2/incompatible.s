// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %ta.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %tb.o
// RUN: lld -flavor gnu2 -shared %tb.o -o %ti686.so
// RUN: llvm-mc -filetype=obj -triple=arm-unknown-linux %s -o %tc.o

// RUN: not lld -flavor gnu2 %ta.o %tb.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-B %s
// A-AND-B: a.o is incompatible with {{.*}}b.o

// RUN: not lld -flavor gnu2 %tb.o %tc.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=B-AND-C %s
// B-AND-C: b.o is incompatible with {{.*}}c.o

// RUN: not lld -flavor gnu2 %ta.o %ti686.so -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-SO %s
// A-AND-SO: a.o is incompatible with {{.*}}i686.so

// RUN: not lld -flavor gnu2 %tc.o %ti686.so -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=C-AND-SO %s
// C-AND-SO: c.o is incompatible with {{.*}}i686.so

// RUN: not lld -flavor gnu2 %ti686.so %tc.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=SO-AND-C %s
// SO-AND-C: i686.so is incompatible with {{.*}}c.o

// REQUIRES: x86,arm
