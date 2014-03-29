// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o /dev/null \
// RUN: 2>&1 | FileCheck %s

// CHECK: Cannot represent a difference across sections

        .long foo - bar
        .section .zed
foo:
        .section .bah
bar:
