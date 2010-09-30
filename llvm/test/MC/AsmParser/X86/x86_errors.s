// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2> %t.err
// RUN: FileCheck < %t.err %s

// CHECK: error: ambiguous instructions require an explicit suffix (could be 'cmpb', 'cmpw', 'cmpl', or 'cmpq')
cmp $0, 0(%eax)
