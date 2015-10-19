# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=micromips -mattr=+dspr2 | FileCheck %s

  .set noat
  absq_s.qb $3, $4             # CHECK: absq_s.qb $3, $4           # encoding: [0x00,0x64,0x01,0x3c]
  dpa.w.ph $ac0, $3, $2        # CHECK: dpa.w.ph $ac0, $3, $2      # encoding: [0x00,0x43,0x00,0xbc]
  dpaqx_s.w.ph $ac3, $12, $7   # CHECK: dpaqx_s.w.ph $ac3, $12, $7 # encoding: [0x00,0xec,0xe2,0xbc]
  dpaqx_sa.w.ph $ac0, $5, $6   # CHECK: dpaqx_sa.w.ph $ac0, $5, $6 # encoding: [0x00,0xc5,0x32,0xbc]
  dpax.w.ph $ac3, $2, $1       # CHECK: dpax.w.ph $ac3, $2, $1     # encoding: [0x00,0x22,0xd0,0xbc]
