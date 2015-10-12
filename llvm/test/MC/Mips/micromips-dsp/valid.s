# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=micromips -mattr=+dsp | FileCheck %s

  .set noat
  addu.qb $3, $4, $5           # CHECK: addu.qb $3, $4, $5      # encoding: [0x00,0xa4,0x18,0xcd]

