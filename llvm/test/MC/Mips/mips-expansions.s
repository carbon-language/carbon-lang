# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | FileCheck %s
# Check that the assembler can handle the documented syntax
# for macro instructions
#------------------------------------------------------------------------------
# Load immediate instructions
#------------------------------------------------------------------------------
# CHECK:     ori     $5, $zero, 123   # encoding: [0x7b,0x00,0x05,0x34]
# CHECK:     addiu   $6, $zero, -2345 # encoding: [0xd7,0xf6,0x06,0x24]
# CHECK:     lui     $7, 1            # encoding: [0x01,0x00,0x07,0x3c]
# CHECK:     ori     $7, $7, 2        # encoding: [0x02,0x00,0xe7,0x34]
# CHECK:     addiu   $8, $zero, -8    # encoding: [0xf8,0xff,0x08,0x24]
# CHECK:     lui     $9, 1            # encoding: [0x01,0x00,0x09,0x3c]
# CHECK-NOT: ori $9, $9, 0            # encoding: [0x00,0x00,0x29,0x35]
# CHECK:     lui     $10, 65519       # encoding: [0xef,0xff,0x0a,0x3c]
# CHECK:     ori     $10, $10, 61423  # encoding: [0xef,0xef,0x4a,0x35]

# CHECK: ori     $4, $zero, 20       # encoding: [0x14,0x00,0x04,0x34]
# CHECK: lui     $7, 1               # encoding: [0x01,0x00,0x07,0x3c]
# CHECK: ori     $7, $7, 2           # encoding: [0x02,0x00,0xe7,0x34]
# CHECK: ori     $4, $5, 20          # encoding: [0x14,0x00,0xa4,0x34]
# CHECK: lui     $7, 1               # encoding: [0x01,0x00,0x07,0x3c]
# CHECK: ori     $7, $7, 2           # encoding: [0x02,0x00,0xe7,0x34]
# CHECK: addu    $7, $7, $8          # encoding: [0x21,0x38,0xe8,0x00]
# CHECK: lui     $8, %hi(symbol)     # encoding: [A,A,0x08,0x3c]
# CHECK:                             #   fixup A - offset: 0, value: symbol@ABS_HI, kind: fixup_Mips_HI16
# CHECK: ori     $8, $8, %lo(symbol) # encoding: [A,A,0x08,0x35]
# CHECK:                             #   fixup A - offset: 0, value: symbol@ABS_LO, kind: fixup_Mips_LO16
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

# CHECK:     lui     $8, %hi(symbol)     # encoding: [A,A,0x08,0x3c]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_HI, kind: fixup_Mips_HI16
# CHECK-NOT: move    $8, $8              # encoding: [0x21,0x40,0x00,0x01]
# CHECK:     lw      $8, %lo(symbol)($8) # encoding: [A,A,0x08,0x8d]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_LO, kind: fixup_Mips_LO16
# CHECK:     lui     $1, %hi(symbol)     # encoding: [A,A,0x01,0x3c]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_HI, kind: fixup_Mips_HI16
# CHECK-NOT: move    $1, $1              # encoding: [0x21,0x08,0x20,0x00]
# CHECK:     sw      $8, %lo(symbol)($1) # encoding: [A,A,0x28,0xac]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_LO, kind: fixup_Mips_LO16

# CHECK: lui     $1, %hi(symbol)
# CHECK: ldc1    $f0, %lo(symbol)($1)
# CHECK: lui     $1, %hi(symbol)
# CHECK: sdc1    $f0, %lo(symbol)($1)

    li $5,123
    li $6,-2345
    li $7,65538
    li $8, ~7
    li $9, 0x10000
    li $10, ~(0x101010)

    la $a0, 20
    la $7,65538
    la $a0, 20($a1)
    la $7,65538($8)
    la $t0, symbol

    .set noat
    lw  $t2, symbol($a0)
    .set at
    sw  $t2, symbol($t1)

    lw  $t2, 655483($a0)
    sw  $t2, 123456($t1)

    lw  $8, symbol
    sw  $8, symbol

    ldc1 $f0, symbol
    sdc1 $f0, symbol
