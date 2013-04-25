# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips | FileCheck %s
# Check that the assembler can handle the documented syntax
# for load and store instructions.
#------------------------------------------------------------------------------
# Load and Store Instructions
#------------------------------------------------------------------------------
# CHECK: lb     $5, 8($4)      # encoding: [0x08,0x00,0xa4,0x1c]
# CHECK: lbu    $6, 8($4)      # encoding: [0x08,0x00,0xc4,0x14]
# CHECK: lh     $2, 8($4)      # encoding: [0x08,0x00,0x44,0x3c]
# CHECK: lhu    $4, 8($2)      # encoding: [0x08,0x00,0x82,0x34]
# CHECK: lw     $6, 4($5)      # encoding: [0x04,0x00,0xc5,0xfc]
# CHECK: sb     $5, 8($4)      # encoding: [0x08,0x00,0xa4,0x18]
# CHECK: sh     $2, 8($4)      # encoding: [0x08,0x00,0x44,0x38]
# CHECK: sw     $5, 4($6)      # encoding: [0x04,0x00,0xa6,0xf8]
     lb     $5, 8($4)
     lbu    $6, 8($4)
     lh     $2, 8($4)
     lhu    $4, 8($2)
     lw     $6, 4($5)
     sb     $5, 8($4)
     sh     $2, 8($4)
     sw     $5, 4($6)
