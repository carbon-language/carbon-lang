# RUN: llvm-mc %s -triple=x86_64-unknown-unknown | FileCheck %s

palignr $8, %xmm0, %xmm1
# CHECK: xmm1 = xmm0[8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4,5,6,7]
palignr $8, (%rax), %xmm1
# CHECK: xmm1 = mem[8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4,5,6,7]

palignr $16, %xmm0, %xmm1
# CHECK: xmm1 = xmm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
palignr $16, (%rax), %xmm1
# CHECK: xmm1 = xmm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

palignr $0, %xmm0, %xmm1
# CHECK: xmm1 = xmm0[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
palignr $0, (%rax), %xmm1
# CHECK: xmm1 = mem[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

vpalignr $8, %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm0[8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4,5,6,7]
vpalignr $8, (%rax), %xmm1, %xmm2
# CHECK: xmm2 = mem[8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4,5,6,7]

vpalignr $16, %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
vpalignr $16, (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

vpalignr $0, %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm0[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
vpalignr $0, (%rax), %xmm1, %xmm2
# CHECK: xmm2 = mem[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
