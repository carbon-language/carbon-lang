# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s

# Check that the IAS expands macro instructions in the same way as GAS.

# Load immediate, done by MipsAsmParser::expandLoadImm():
  li $5, 123
# CHECK:     ori     $5, $zero, 123   # encoding: [0x7b,0x00,0x05,0x34]
  li $6, -2345
# CHECK:     addiu   $6, $zero, -2345 # encoding: [0xd7,0xf6,0x06,0x24]
  li $7, 65538
# CHECK:     lui     $7, 1            # encoding: [0x01,0x00,0x07,0x3c]
# CHECK:     ori     $7, $7, 2        # encoding: [0x02,0x00,0xe7,0x34]
  li $8, ~7
# CHECK:     addiu   $8, $zero, -8    # encoding: [0xf8,0xff,0x08,0x24]
  li $9, 0x10000
# CHECK:     lui     $9, 1            # encoding: [0x01,0x00,0x09,0x3c]
# CHECK-NOT: ori $9, $9, 0            # encoding: [0x00,0x00,0x29,0x35]
  li $10, ~(0x101010)
# CHECK:     lui     $10, 65519       # encoding: [0xef,0xff,0x0a,0x3c]
# CHECK:     ori     $10, $10, 61423  # encoding: [0xef,0xef,0x4a,0x35]

# Load address, done by MipsAsmParser::expandLoadAddressReg()
# and MipsAsmParser::expandLoadAddressImm():
  la $4, 20
# CHECK: ori     $4, $zero, 20       # encoding: [0x14,0x00,0x04,0x34]
  la $7, 65538
# CHECK: lui     $7, 1               # encoding: [0x01,0x00,0x07,0x3c]
# CHECK: ori     $7, $7, 2           # encoding: [0x02,0x00,0xe7,0x34]
  la $4, 20($5)
# CHECK: ori     $4, $5, 20          # encoding: [0x14,0x00,0xa4,0x34]
  la $7, 65538($8)
# CHECK: lui     $7, 1               # encoding: [0x01,0x00,0x07,0x3c]
# CHECK: ori     $7, $7, 2           # encoding: [0x02,0x00,0xe7,0x34]
# CHECK: addu    $7, $7, $8          # encoding: [0x21,0x38,0xe8,0x00]
  la $8, symbol
# CHECK: lui     $8, %hi(symbol)     # encoding: [A,A,0x08,0x3c]
# CHECK:                             #   fixup A - offset: 0, value: symbol@ABS_HI, kind: fixup_Mips_HI16
# CHECK: ori     $8, $8, %lo(symbol) # encoding: [A,A,0x08,0x35]
# CHECK:                             #   fixup A - offset: 0, value: symbol@ABS_LO, kind: fixup_Mips_LO16

# LW/SW and LDC1/SDC1 of symbol address, done by MipsAsmParser::expandMemInst():
  .set noat
  lw $10, symbol($4)
# CHECK: lui     $10, %hi(symbol)        # encoding: [A,A,0x0a,0x3c]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_HI, kind: fixup_Mips_HI16
# CHECK: addu    $10, $10, $4            # encoding: [0x21,0x50,0x44,0x01]
# CHECK: lw      $10, %lo(symbol)($10)   # encoding: [A,A,0x4a,0x8d]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_LO, kind: fixup_Mips_LO16
  .set at
  sw $10, symbol($9)
# CHECK: lui     $1, %hi(symbol)         # encoding: [A,A,0x01,0x3c]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_HI, kind: fixup_Mips_HI16
# CHECK: addu    $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
# CHECK: sw      $10, %lo(symbol)($1)    # encoding: [A,A,0x2a,0xac]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_LO, kind: fixup_Mips_LO16

  lw $8, 1f
# CHECK: lui $8, %hi($tmp0)              # encoding: [A,A,0x08,0x3c]
# CHECK:                                 #   fixup A - offset: 0, value: ($tmp0)@ABS_HI, kind: fixup_Mips_HI16
# CHECK: lw  $8, %lo($tmp0)($8)          # encoding: [A,A,0x08,0x8d]
# CHECK:                                 #   fixup A - offset: 0, value: ($tmp0)@ABS_LO, kind: fixup_Mips_LO16

  lw $10, 655483($4)
# CHECK: lui     $10, 10                 # encoding: [0x0a,0x00,0x0a,0x3c]
# CHECK: addu    $10, $10, $4            # encoding: [0x21,0x50,0x44,0x01]
# CHECK: lw      $10, 123($10)           # encoding: [0x7b,0x00,0x4a,0x8d]
  sw $10, 123456($9)
# CHECK: lui     $1, 2                   # encoding: [0x02,0x00,0x01,0x3c]
# CHECK: addu    $1, $1, $9              # encoding: [0x21,0x08,0x29,0x00]
# CHECK: sw      $10, 57920($1)          # encoding: [0x40,0xe2,0x2a,0xac]

  lw $8, symbol
# CHECK:     lui     $8, %hi(symbol)     # encoding: [A,A,0x08,0x3c]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_HI, kind: fixup_Mips_HI16
# CHECK-NOT: move    $8, $8              # encoding: [0x21,0x40,0x00,0x01]
# CHECK:     lw      $8, %lo(symbol)($8) # encoding: [A,A,0x08,0x8d]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_LO, kind: fixup_Mips_LO16
  sw $8, symbol
# CHECK:     lui     $1, %hi(symbol)     # encoding: [A,A,0x01,0x3c]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_HI, kind: fixup_Mips_HI16
# CHECK-NOT: move    $1, $1              # encoding: [0x21,0x08,0x20,0x00]
# CHECK:     sw      $8, %lo(symbol)($1) # encoding: [A,A,0x28,0xac]
# CHECK:                                 #   fixup A - offset: 0, value: symbol@ABS_LO, kind: fixup_Mips_LO16

  ldc1 $f0, symbol
# CHECK: lui     $1, %hi(symbol)
# CHECK: ldc1    $f0, %lo(symbol)($1)
  sdc1 $f0, symbol
# CHECK: lui     $1, %hi(symbol)
# CHECK: sdc1    $f0, %lo(symbol)($1)

# Test BNE with an immediate as the 2nd operand.
  bne $2, 0, 1332
# CHECK: bnez  $2, 1332          # encoding: [0x4d,0x01,0x40,0x14]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

  bne $2, 123, 1332
# CHECK: ori   $1, $zero, 123    # encoding: [0x7b,0x00,0x01,0x34]
# CHECK: bne   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x14]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

  bne $2, -2345, 1332
# CHECK: addiu $1, $zero, -2345  # encoding: [0xd7,0xf6,0x01,0x24]
# CHECK: bne   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x14]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

  bne $2, 65538, 1332
# CHECK: lui   $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK: ori   $1, $1, 2         # encoding: [0x02,0x00,0x21,0x34]
# CHECK: bne   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x14]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

  bne $2, ~7, 1332
# CHECK: addiu $1, $zero, -8     # encoding: [0xf8,0xff,0x01,0x24]
# CHECK: bne   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x14]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

  bne $2, 0x10000, 1332
# CHECK: lui   $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK: bne   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x14]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

# Test BEQ with an immediate as the 2nd operand.
  beq $2, 0, 1332
# CHECK: beqz  $2, 1332          # encoding: [0x4d,0x01,0x40,0x10]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, 123, 1332
# CHECK: ori   $1, $zero, 123    # encoding: [0x7b,0x00,0x01,0x34]
# CHECK: beq   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x10]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, -2345, 1332
# CHECK: addiu $1, $zero, -2345  # encoding: [0xd7,0xf6,0x01,0x24]
# CHECK: beq   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x10]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, 65538, 1332
# CHECK: lui   $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK: ori   $1, $1, 2         # encoding: [0x02,0x00,0x21,0x34]
# CHECK: beq   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x10]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, ~7, 1332
# CHECK: addiu $1, $zero, -8     # encoding: [0xf8,0xff,0x01,0x24]
# CHECK: beq   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x10]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

  beq $2, 0x10000, 1332
# CHECK: lui   $1, 1             # encoding: [0x01,0x00,0x01,0x3c]
# CHECK: beq   $2, $1, 1332      # encoding: [0x4d,0x01,0x41,0x10]
# CHECK: nop                     # encoding: [0x00,0x00,0x00,0x00]

1:
  add $4, $4, $4
