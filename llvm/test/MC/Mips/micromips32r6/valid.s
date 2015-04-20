# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=micromips | FileCheck %s

  .set noat
  balc 14572256            # CHECK: balc 14572256       # encoding: [0xb4,0x37,0x96,0xb8]
  bc 14572256              # CHECK: bc 14572256         # encoding: [0x94,0x37,0x96,0xb8]
