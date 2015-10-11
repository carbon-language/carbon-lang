# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-freebsd %s -o %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-freebsd \
# RUN:   %p/Inputs/libsearch-dyn.s -o %tdyn.o
# RUN: ld.lld2 -shared %tdyn.o -o %T/libls.so
# RUN: echo "SEARCH_DIR(" %T ")" > %t.script
# RUN: ld.lld2 -o %t2 --script %t.script -lls %t

.globl _start,_bar
_start:
