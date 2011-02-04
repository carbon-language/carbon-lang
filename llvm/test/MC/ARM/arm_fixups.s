// RUN: llvm-mc -triple arm-unknown-unknown %s --show-encoding > %t
// RUN: FileCheck < %t %s

// CHECK: bl _printf @ encoding: [A,A,A,0xeb]
// CHECK: @ fixup A - offset: 0, value: _printf, kind: fixup_arm_uncondbranch
bl _printf
        