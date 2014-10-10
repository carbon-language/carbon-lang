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
