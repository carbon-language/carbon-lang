# RUN: llvm-mc %s -triple=mipsel -show-encoding -mcpu=mips32r2 -mattr=micromips | FileCheck %s
# Check that the assembler can handle the documented syntax
# for shift instructions.
#------------------------------------------------------------------------------
# Shift Instructions
#------------------------------------------------------------------------------
# CHECK: sll    $4, $3, 7      # encoding: [0x00,0x38,0x83,0x00]
# CHECK: sllv   $2, $3, $5     # encoding: [0x10,0x10,0x65,0x00]
# CHECK: sra    $4, $3, 7      # encoding: [0x80,0x38,0x83,0x00]
# CHECK: srav   $2, $3, $5     # encoding: [0x90,0x10,0x65,0x00]
# CHECK: srl    $4, $3, 7      # encoding: [0x40,0x38,0x83,0x00]
# CHECK: srlv   $2, $3, $5     # encoding: [0x50,0x10,0x65,0x00]
# CHECK: rotr   $9, $6, 7      # encoding: [0xc0,0x38,0x26,0x01]
# CHECK: rotrv  $9, $6, $7     # encoding: [0xd0,0x48,0xc7,0x00]
     sll    $4, $3, 7
     sllv   $2, $3, $5
     sra    $4, $3, 7
     srav   $2, $3, $5
     srl    $4, $3, 7
     srlv   $2, $3, $5
     rotr   $9, $6, 7
     rotrv  $9, $6, $7
