// RUN: llvm-mc -triple aarch64--none-eabi -filetype obj < %s -o - | llvm-objdump -d - | FileCheck %s

// CHECK: mov     x0, #1311673391471656960
movz x0, #:abs_g3:fourpart
// CHECK: mov     x0, #20014547599360
movz x0, #:abs_g2:threepart
// CHECK: movk    x0, #22136, lsl #32
movk x0, #:abs_g2_nc:fourpart
// CHECK: mov     x0, #305397760
movz x0, #:abs_g1:twopart
// CHECK: movk    x0, #37035, lsl #16
movk x0, #:abs_g1_nc:fourpart
// CHECK: mov     x0, #4660
movz x0, #:abs_g0:onepart
// CHECK: movk    x0, #52719
movk x0, #:abs_g0_nc:fourpart

onepart = 0x1234
twopart = 0x12345678
threepart = 0x1234567890AB
fourpart = 0x1234567890ABCDEF
