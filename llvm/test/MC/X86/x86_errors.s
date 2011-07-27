// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2> %t.err
// RUN: FileCheck --check-prefix=64 < %t.err %s

// RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t.err
// RUN: FileCheck --check-prefix=32 < %t.err %s
// rdar://8204588

// 64: error: ambiguous instructions require an explicit suffix (could be 'cmpb', 'cmpw', 'cmpl', or 'cmpq')
cmp $0, 0(%eax)

// 32: error: register %rax is only available in 64-bit mode
addl $0, 0(%rax)
