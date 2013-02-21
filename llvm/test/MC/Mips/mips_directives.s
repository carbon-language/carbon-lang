# RUN: llvm-mc -show-encoding -triple mips-unknown-unknown %s | FileCheck %s
#
$BB0_2:
  .ent directives_test
    .frame    $sp,0,$ra
    .mask     0x00000000,0
    .fmask    0x00000000,0
    .set    noreorder
    .set    nomacro
    .set    noat
$JTI0_0:
    .gpword    ($BB0_2)
    .word 0x77fffffc
# CHECK: $JTI0_0:
# CHECK-NEXT:     .4byte    2013265916
    .set  at=$12
    .set macro
    .set reorder
    .set  at=$a0
