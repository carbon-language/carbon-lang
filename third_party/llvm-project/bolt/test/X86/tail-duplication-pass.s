# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --data %t.fdata --reorder-blocks=ext-tsp \
# RUN:    --print-finalized --tail-duplication=moderate \
# RUN:    --tail-duplication-minimum-offset=1 -o %t.out | FileCheck %s

# FDATA: 1 main 2 1 main #.BB2# 0 10
# FDATA: 1 main 4 1 main #.BB2# 0 20
# CHECK: BOLT-INFO: tail duplication modified 1 ({{.*}}%) functions; duplicated 1 blocks (1 bytes) responsible for {{.*}} dynamic executions ({{.*}}% of all block executions)
# CHECK: BB Layout   : .LBB00, .Ltail-dup0, .Ltmp0, .Ltmp1

    .text
    .globl main
    .type main, %function
    .size main, .Lend-main
main:
    xor %eax, %eax
    jmp .BB2
.BB1:
    inc %rax
.BB2:
    retq
# For relocations against .text
    call exit
.Lend:
