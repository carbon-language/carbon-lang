// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
// RUN:   %p/Inputs/libsearch-dyn.s -o %tdyn.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
// RUN:   %p/Inputs/libsearch-st.s -o %tst.o
// RUN: lld -flavor gnu2 -shared %tdyn.o -o %T/libls.so
// RUN: cp -f %T/libls.so %T/libls2.so
// RUN: rm -f %T/libls.a
// RUN: llvm-ar rcs %T/libls.a %tst.o
// REQUIRES: x86

// Should not link because of undefined symbol _bar
// RUN: not lld -flavor gnu2 -o %t3 %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=UNDEFINED %s
// UNDEFINED: undefined symbol: _bar

// Should fail if cannot find specified library (without -L switch)
// RUN: not lld -flavor gnu2 -o %t3 %t.o -lls 2>&1 \
// RUN:   | FileCheck --check-prefix=NOLIB %s
// NOLIB: Unable to find library -lls

// Should use explicitly specified static library
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -l:libls.a
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s
// STATIC: Symbols [
// STATIC: Name: _static
// STATIC: ]

// Should use explicitly specified dynamic library
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -l:libls.so
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s
// DYNAMIC: Symbols [
// DYNAMIC-NOT: Name: _static
// DYNAMIC: ]

// Should prefer dynamic to static
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s

// -L can be placed after -l
// RUN: lld -flavor gnu2 -o %t3 %t.o -lls -L%T

// Check long forms as well
// RUN: lld -flavor gnu2 -o %t3 %t.o --library-path=%T --library=ls

// Should not search for dynamic libraries if -Bstatic is specified
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -Bstatic -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s
// RUN: not lld -flavor gnu2 -o %t3 %t.o -L%T -Bstatic -lls2 2>&1 \
// RUN:   | FileCheck --check-prefix=NOLIB2 %s
// NOLIB2: Unable to find library -lls2

// -Bdynamic should restore default behaviour
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -Bstatic -Bdynamic -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s

// -Bstatic and -Bdynamic should affect only libraries which follow them
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -lls -Bstatic -Bdynamic
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -Bstatic -lls -Bdynamic
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s

// Check aliases as well
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -dn -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -non_shared -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -static -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -Bstatic -dy -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s
// RUN: lld -flavor gnu2 -o %t3 %t.o -L%T -Bstatic -call_shared -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s

.globl _start,_bar;
_start:
