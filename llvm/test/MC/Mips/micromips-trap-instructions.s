# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for miscellaneous instructions
#------------------------------------------------------------------------------
# Miscellaneous Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: teq     $8, $9, 0    # encoding: [0x28,0x01,0x3c,0x00]
# CHECK-EL: tge     $8, $9, 0    # encoding: [0x28,0x01,0x3c,0x02]
# CHECK-EL: tgeu    $8, $9, 0    # encoding: [0x28,0x01,0x3c,0x04]
# CHECK-EL: tlt     $8, $9, 0    # encoding: [0x28,0x01,0x3c,0x08]
# CHECK-EL: tltu    $8, $9, 0    # encoding: [0x28,0x01,0x3c,0x0a]
# CHECK-EL: tne     $8, $9, 0    # encoding: [0x28,0x01,0x3c,0x0c]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: teq     $8, $9, 0    # encoding: [0x01,0x28,0x00,0x3c]
# CHECK-EB: tge     $8, $9, 0    # encoding: [0x01,0x28,0x02,0x3c]
# CHECK-EB: tgeu    $8, $9, 0    # encoding: [0x01,0x28,0x04,0x3c]
# CHECK-EB: tlt     $8, $9, 0    # encoding: [0x01,0x28,0x08,0x3c]
# CHECK-EB: tltu    $8, $9, 0    # encoding: [0x01,0x28,0x0a,0x3c]
# CHECK-EB: tne     $8, $9, 0    # encoding: [0x01,0x28,0x0c,0x3c]
    teq     $8, $9, 0
    tge     $8, $9, 0
    tgeu    $8, $9, 0
    tlt     $8, $9, 0
    tltu    $8, $9, 0
    tne     $8, $9, 0
