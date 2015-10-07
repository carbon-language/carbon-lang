// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %ta.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %tb.o
// RUN: ld.lld2 -shared %tb.o -o %ti686.so
// RUN: llvm-mc -filetype=obj -triple=arm-unknown-linux %s -o %tc.o

// RUN: not lld -flavor gnu2 %ta.o %tb.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-B %s
// A-AND-B: b.o is incompatible with {{.*}}a.o

// RUN: not lld -flavor gnu2 %tb.o %tc.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=B-AND-C %s
// B-AND-C: c.o is incompatible with {{.*}}b.o

// RUN: not lld -flavor gnu2 %ta.o %ti686.so -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=A-AND-SO %s
// A-AND-SO: i686.so is incompatible with {{.*}}a.o

// RUN: not lld -flavor gnu2 %tc.o %ti686.so -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=C-AND-SO %s
// C-AND-SO: i686.so is incompatible with {{.*}}c.o

// RUN: not lld -flavor gnu2 %ti686.so %tc.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=SO-AND-C %s
// SO-AND-C: c.o is incompatible with {{.*}}i686.so

// RUN: not lld -flavor gnu2 -m elf64ppc %ta.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=A-ONLY %s
// A-ONLY: a.o is incompatible with target architecture

// RUN: not lld -flavor gnu2 -m elf64ppc %tb.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=B-ONLY %s
// B-ONLY: b.o is incompatible with target architecture

// RUN: not lld -flavor gnu2 -m elf64ppc %tc.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=C-ONLY %s
// C-ONLY: c.o is incompatible with target architecture

// RUN: not lld -flavor gnu2 -m elf_i386 %tc.o %ti686.so -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=C-AND-SO-I386 %s
// C-AND-SO-I386: c.o is incompatible with target architecture

// RUN: not lld -flavor gnu2 -m elf_i386 %ti686.so %tc.o -o %t 2>&1 | \
// RUN:   FileCheck --check-prefix=SO-AND-C-I386 %s
// SO-AND-C-I386: c.o is incompatible with {{.*}}i686.so

// REQUIRES: x86,arm
