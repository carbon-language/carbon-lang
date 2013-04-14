# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | FileCheck %s
# Check that the assembler can handle the documented syntax
# for macro instructions
#------------------------------------------------------------------------------
# Load immediate instructions
#------------------------------------------------------------------------------
# CHECK: ori     $5, $zero, 123      # encoding: [0x7b,0x00,0x05,0x34]
# CHECK: addiu   $6, $zero, -2345    # encoding: [0xd7,0xf6,0x06,0x24]
# CHECK: lui     $7, 1               # encoding: [0x01,0x00,0x07,0x3c]
# CHECK: ori     $7, $7, 2           # encoding: [0x02,0x00,0xe7,0x34]
# CHECK: addiu   $4, $zero, 20       # encoding: [0x14,0x00,0x04,0x24]
# CHECK: lui     $7, 1               # encoding: [0x01,0x00,0x07,0x3c]
# CHECK: ori     $7, $7, 2           # encoding: [0x02,0x00,0xe7,0x34]
# CHECK: addiu   $4, $5, 20          # encoding: [0x14,0x00,0xa4,0x24]
# CHECK: lui     $7, 1               # encoding: [0x01,0x00,0x07,0x3c]
# CHECK: ori     $7, $7, 2           # encoding: [0x02,0x00,0xe7,0x34]
# CHECK: addu    $7, $7, $8          # encoding: [0x21,0x38,0xe8,0x00]
# CHECK: lui     $10, %hi(symbol)        # encoding: [A,A,0x0a,0x3c]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_HI, kind: fixup_Mips_HI16
# CHECK: addu    $10, $10, $4            # encoding: [0x21,0x50,0x44,0x01]
# CHECK: lw      $10, %lo(symbol)($10)   # encoding: [A,A,0x4a,0x8d]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_LO, kind: fixup_Mips_LO16
# CHECK: lui     $1, %hi(symbol)         # encoding: [A,A,0x01,0x3c]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_HI, kind: fixup_Mips_HI16
# CHECK: addu    $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
# CHECK: sw      $10, %lo(symbol)($1)    # encoding: [A,A,0x2a,0xac]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_LO, kind: fixup_Mips_LO16
# CHECK: lui     $10, 10                 # encoding: [0x0a,0x00,0x0a,0x3c]
# CHECK: addu    $10, $10, $4            # encoding: [0x21,0x50,0x44,0x01]
# CHECK: lw      $10, 123($10)           # encoding: [0x7b,0x00,0x4a,0x8d]
# CHECK: lui     $1, 2                   # encoding: [0x02,0x00,0x01,0x3c]
# CHECK: addu    $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
# CHECK: sw      $10, 57920($1)          # encoding: [0x40,0xe2,0x2a,0xac]

    li $5,123
    li $6,-2345
    li $7,65538

    la $a0, 20
    la $7,65538
    la $a0, 20($a1)
    la $7,65538($8)

    lw  $t2, symbol($a0)
    sw  $t2, symbol($t1)

    lw  $t2, 655483($a0)
    sw  $t2, 123456($t1)
