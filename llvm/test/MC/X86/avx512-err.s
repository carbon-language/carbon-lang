// RUN: not llvm-mc -triple x86_64-unknown-unknown -mcpu=knl -mattr=+avx512dq --show-encoding %s 2> %t.err
// RUN: FileCheck --check-prefix=ERR < %t.err %s

// ERR: invalid operand for instruction
vpcmpd $1, %zmm24, %zmm7, %k5{%k0}

