// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o /dev/null \
// RUN: 2>&1 | FileCheck %s

// CHECK: No relocation available to represent this relative expression
// CHECK: call foo - bar


        call foo - bar
        .section .foo
foo:
        .section .bar
bar:
