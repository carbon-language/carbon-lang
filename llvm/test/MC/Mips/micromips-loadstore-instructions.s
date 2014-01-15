# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for load and store instructions.
#------------------------------------------------------------------------------
# Load and Store Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: lb     $5, 8($4)      # encoding: [0xa4,0x1c,0x08,0x00]
# CHECK-EL: lbu    $6, 8($4)      # encoding: [0xc4,0x14,0x08,0x00]
# CHECK-EL: lh     $2, 8($4)      # encoding: [0x44,0x3c,0x08,0x00]
# CHECK-EL: lhu    $4, 8($2)      # encoding: [0x82,0x34,0x08,0x00]
# CHECK-EL: lw     $6, 4($5)      # encoding: [0xc5,0xfc,0x04,0x00]
# CHECK-EL: sb     $5, 8($4)      # encoding: [0xa4,0x18,0x08,0x00]
# CHECK-EL: sh     $2, 8($4)      # encoding: [0x44,0x38,0x08,0x00]
# CHECK-EL: sw     $5, 4($6)      # encoding: [0xa6,0xf8,0x04,0x00]
# CHECK-EL: ll     $2, 8($4)      # encoding: [0x44,0x60,0x08,0x30]
# CHECK-EL: sc     $2, 8($4)      # encoding: [0x44,0x60,0x08,0xb0]
# CHECK-EL: lwu    $2, 8($4)      # encoding: [0x44,0x60,0x08,0xe0]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: lb     $5, 8($4)      # encoding: [0x1c,0xa4,0x00,0x08]
# CHECK-EB: lbu    $6, 8($4)      # encoding: [0x14,0xc4,0x00,0x08]
# CHECK-EB: lh     $2, 8($4)      # encoding: [0x3c,0x44,0x00,0x08]
# CHECK-EB: lhu    $4, 8($2)      # encoding: [0x34,0x82,0x00,0x08]
# CHECK-EB: lw     $6, 4($5)      # encoding: [0xfc,0xc5,0x00,0x04]
# CHECK-EB: sb     $5, 8($4)      # encoding: [0x18,0xa4,0x00,0x08]
# CHECK-EB: sh     $2, 8($4)      # encoding: [0x38,0x44,0x00,0x08]
# CHECK-EB: sw     $5, 4($6)      # encoding: [0xf8,0xa6,0x00,0x04]
# CHECK-EB: ll     $2, 8($4)      # encoding: [0x60,0x44,0x30,0x08]
# CHECK-EB: sc     $2, 8($4)      # encoding: [0x60,0x44,0xb0,0x08]
# CHECK-EB: lwu    $2, 8($4)      # encoding: [0x60,0x44,0xe0,0x08]
     lb     $5, 8($4)
     lbu    $6, 8($4)
     lh     $2, 8($4)
     lhu    $4, 8($2)
     lw     $6, 4($5)
     sb     $5, 8($4)
     sh     $2, 8($4)
     sw     $5, 4($6)
     ll     $2, 8($4)
     sc     $2, 8($4)
     lwu    $2, 8($4)
