// RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t
// RUN: FileCheck --input-file %t %s

        .data
// CHECK: invalid assignment to 't0_v0'
        t0_v0 = t0_v0 + 1

        t1_v1 = 1
        t1_v1 = 2

t2_s0:
// CHECK: redefinition of 't2_s0'
        t2_s0 = 2

        t3_s0 = t2_s0 + 1
// CHECK: invalid reassignment of non-absolute variable 't3_s0'
        t3_s0 = 1
