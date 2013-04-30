# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips | FileCheck %s
# Check that the assembler can handle the documented syntax
# for arithmetic and logical instructions.
#------------------------------------------------------------------------------
# Arithmetic and Logical Instructions
#------------------------------------------------------------------------------
# CHECK: add   $9, $6, $7      # encoding: [0x10,0x49,0xe6,0x00]
# CHECK: addi  $9, $6, 17767   # encoding: [0x67,0x45,0x26,0x11]
# CHECK: addiu $9, $6, -15001  # encoding: [0x67,0xc5,0x26,0x31]
# CHECK: addi  $9, $6, 17767   # encoding: [0x67,0x45,0x26,0x11]
# CHECK: addiu $9, $6, -15001  # encoding: [0x67,0xc5,0x26,0x31]
# CHECK: addu  $9, $6, $7      # encoding: [0x50,0x49,0xe6,0x00]
# CHECK: sub   $9, $6, $7      # encoding: [0x90,0x49,0xe6,0x00]
# CHECK: subu  $4, $3, $5      # encoding: [0xd0,0x21,0xa3,0x00]
# CHECK: neg   $6, $7          # encoding: [0x90,0x31,0xe0,0x00]
# CHECK: negu  $6, $7          # encoding: [0xd0,0x31,0xe0,0x00]
# CHECK: move  $7, $8          # encoding: [0x50,0x39,0x08,0x00]
# CHECK: slt    $3, $3, $5     # encoding: [0x50,0x1b,0xa3,0x00]
# CHECK: slti   $3, $3, 103    # encoding: [0x67,0x00,0x63,0x90]
# CHECK: slti   $3, $3, 103    # encoding: [0x67,0x00,0x63,0x90]
# CHECK: sltiu  $3, $3, 103    # encoding: [0x67,0x00,0x63,0xb0]
# CHECK: sltu   $3, $3, $5     # encoding: [0x90,0x1b,0xa3,0x00]
# CHECK: and    $9, $6, $7     # encoding: [0x50,0x4a,0xe6,0x00]
# CHECK: andi   $9, $6, 17767  # encoding: [0x67,0x45,0x26,0xd1]
# CHECK: andi   $9, $6, 17767  # encoding: [0x67,0x45,0x26,0xd1]
# CHECK: or     $3, $4, $5     # encoding: [0x90,0x1a,0xa4,0x00]
# CHECK: ori    $9, $6, 17767  # encoding: [0x67,0x45,0x26,0x51]
# CHECK: xor    $3, $3, $5     # encoding: [0x10,0x1b,0xa3,0x00]
# CHECK: xori   $9, $6, 17767  # encoding: [0x67,0x45,0x26,0x71]
# CHECK: xori   $9, $6, 17767  # encoding: [0x67,0x45,0x26,0x71]
# CHECK: nor    $9, $6, $7     # encoding: [0xd0,0x4a,0xe6,0x00]
# CHECK: not    $7, $8         # encoding: [0xd0,0x3a,0x08,0x00]
# CHECK: mul    $9, $6, $7     # encoding: [0x10,0x4a,0xe6,0x00]
# CHECK: mult   $9, $7         # encoding: [0x3c,0x8b,0xe9,0x00]
# CHECK: multu  $9, $7         # encoding: [0x3c,0x9b,0xe9,0x00]
    add    $9, $6, $7
    add    $9, $6, 17767
    addu   $9, $6, -15001
    addi   $9, $6, 17767
    addiu  $9, $6,-15001
    addu   $9, $6, $7
    sub    $9, $6, $7
    subu   $4, $3, $5
    neg    $6, $7
    negu   $6, $7
    move   $7, $8
    slt    $3, $3, $5
    slt    $3, $3, 103
    slti   $3, $3, 103
    sltiu  $3, $3, 103
    sltu   $3, $3, $5
    and    $9, $6, $7
    and    $9, $6, 17767
    andi   $9, $6, 17767
    or     $3, $4, $5
    ori    $9, $6, 17767
    xor    $3, $3, $5
    xor    $9, $6, 17767
    xori   $9, $6, 17767
    nor    $9, $6, $7
    nor    $7, $8, $zero
    mul    $9, $6, $7
    mult   $9, $7
    multu  $9, $7
