// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o -relax-relocations
// RUN: ld.lld -shared %t.o -o %t.so

        movl bar@GOT(%ebx), %eax
