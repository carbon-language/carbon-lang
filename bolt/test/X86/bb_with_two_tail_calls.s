# This reproduces a bug with dynostats when trying to compute branch stats
# at a block with two tails calls (one conditional and one unconditional).

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: strip --strip-unneeded %t.o
# RUN: %host_cc %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out -data %t.fdata -lite=0 -dyno-stats \
# RUN:    -print-sctc 2>&1 | FileCheck %s
# CHECK-NOT: Assertion `BranchInfo.size() == 2 && "could only be called for blocks with 2 successors"' failed.
# Two tail calls in the same basic block after SCTC:
# CHECK:         {{.*}}:   jae     {{.*}} # TAILCALL  # CTCTakenCount: {{.*}}
# CHECK-NEXT:    {{.*}}:   jmp     {{.*}} # TAILCALL

  .globl _start
_start:
    ja a
b:  jb c
# FDATA: 1 _start #b# 1 _start #c# 2 4
    jmp e
a:  nop
c:  jmp e

  .globl e
e:
    nop
