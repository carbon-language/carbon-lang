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
# CHECK-EL: move    $25, $1         # encoding: [0x21,0x0f]
# CHECK-EL: jalr    $9              # encoding: [0xc9,0x45]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: move    $25, $1         # encoding: [0x0f,0x21]
# CHECK-EB: jalr    $9              # encoding: [0x45,0xc9]

    move    $25, $1
    jalr    $9
