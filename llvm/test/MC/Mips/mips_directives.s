# RUN: llvm-mc -show-encoding -triple mips-unknown-unknown %s | FileCheck %s
#
# CHECK:  .text
# CHECK:  $BB0_2:
$BB0_2:
  .ent directives_test
    .frame    $sp,0,$ra
    .mask     0x00000000,0
    .fmask    0x00000000,0
# CHECK:   b 1332               # encoding: [0x10,0x00,0x01,0x4d]
# CHECK:   j 1328               # encoding: [0x08,0x00,0x01,0x4c]
# CHECK:   jal 1328             # encoding: [0x0c,0x00,0x01,0x4c]

    .set    noreorder
     b 1332
     j 1328
     jal 1328
    .set    nomacro
    .set    noat
$JTI0_0:
    .gpword    ($BB0_2)
    .word 0x77fffffc
# CHECK: $JTI0_0:
# CHECK-NEXT:     .4byte    2013265916
    .set  at=$12
    .set macro
# CHECK:   b 1332               # encoding: [0x10,0x00,0x01,0x4d]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   j 1328               # encoding: [0x08,0x00,0x01,0x4c]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jal 1328             # encoding: [0x0c,0x00,0x01,0x4c]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
    .set reorder
    b 1332
    j 1328
    jal 1328
    .set  at=$a0
    .set STORE_MASK,$t7
    .set FPU_MASK,$f7
    .set r3,$3
    .set f6,$f6
#CHECK:    abs.s   $f6, $f7           # encoding: [0x46,0x00,0x39,0x85]
#CHECK:    and     $3, $15, $15       # encoding: [0x01,0xef,0x18,0x24]
    abs.s  f6,FPU_MASK
    and    r3,$t7,STORE_MASK
