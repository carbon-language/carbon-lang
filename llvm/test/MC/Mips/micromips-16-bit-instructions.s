# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips | \
# RUN: FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips | \
# RUN: FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for arithmetic and logical instructions.
#------------------------------------------------------------------------------
# MicroMIPS 16-bit Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: addu16  $6, $17, $4     # encoding: [0x42,0x07]
# CHECK-EL: subu16  $5, $16, $3     # encoding: [0xb1,0x06]
# CHECK-EL: and16   $16, $2         # encoding: [0x82,0x44]
# CHECK-EL: not16   $17, $3         # encoding: [0x0b,0x44]
# CHECK-EL: or16    $16, $4         # encoding: [0xc4,0x44]
# CHECK-EL: xor16   $17, $5         # encoding: [0x4d,0x44]
# CHECK-EL: addius5 $7, -2          # encoding: [0xfc,0x4c]
# CHECK-EL: addiusp -16             # encoding: [0xf9,0x4f]
# CHECK-EL: mfhi    $9              # encoding: [0x09,0x46]
# CHECK-EL: mflo    $9              # encoding: [0x49,0x46]
# CHECK-EL: move    $25, $1         # encoding: [0x21,0x0f]
# CHECK-EL: jrc     $9              # encoding: [0xa9,0x45]
# CHECK-NEXT: jalr    $9            # encoding: [0xc9,0x45]
# CHECK-EL: jraddiusp 20            # encoding: [0x05,0x47]
# CHECK-EL: nop                     # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: jalrs16 $9              # encoding: [0xe9,0x45]
# CHECK-EL: move    $zero, $zero    # encoding: [0x00,0x0c]
# CHECK-EL: jr16    $9              # encoding: [0x89,0x45]
# CHECK-EL: nop                     # encoding: [0x00,0x00,0x00,0x00]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: addu16  $6, $17, $4     # encoding: [0x07,0x42]
# CHECK-EB: subu16  $5, $16, $3     # encoding: [0x06,0xb1]
# CHECK-EB: and16   $16, $2         # encoding: [0x44,0x82]
# CHECK-EB: not16   $17, $3         # encoding: [0x44,0x0b]
# CHECK-EB: or16    $16, $4         # encoding: [0x44,0xc4]
# CHECK-EB: xor16   $17, $5         # encoding: [0x44,0x4d]
# CHECK-EB: addius5 $7, -2          # encoding: [0x4c,0xfc]
# CHECK-EB: addiusp -16             # encoding: [0x4f,0xf9]
# CHECK-EB: mfhi    $9              # encoding: [0x46,0x09]
# CHECK-EB: mflo    $9              # encoding: [0x46,0x49]
# CHECK-EB: move    $25, $1         # encoding: [0x0f,0x21]
# CHECK-EB: jrc     $9              # encoding: [0x45,0xa9]
# CHECK-NEXT: jalr    $9            # encoding: [0x45,0xc9]
# CHECK-EB: jraddiusp 20            # encoding: [0x47,0x05]
# CHECK-EB: nop                     # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: jalrs16 $9              # encoding: [0x45,0xe9]
# CHECK-EB: move    $zero, $zero    # encoding: [0x0c,0x00]
# CHECK-EB: jr16    $9              # encoding: [0x45,0x89]
# CHECK-EB: nop                     # encoding: [0x00,0x00,0x00,0x00]

    addu16  $6, $17, $4
    subu16  $5, $16, $3
    and16   $16, $2
    not16   $17, $3
    or16    $16, $4
    xor16   $17, $5
    addius5 $7, -2
    addiusp -16
    mfhi    $9
    mflo    $9
    move    $25, $1
    jrc     $9
    jalr    $9
    jraddiusp 20
    jalrs16 $9
    jr16    $9
