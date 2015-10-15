# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=micromips -mattr=+dsp | FileCheck %s

  .set noat
  addu.qb $3, $4, $5           # CHECK: addu.qb $3, $4, $5      # encoding: [0x00,0xa4,0x18,0xcd]
  dpaq_s.w.ph $ac1, $5, $3     # CHECK: dpaq_s.w.ph $ac1, $5, $3   # encoding: [0x00,0x65,0x42,0xbc]
  dpaq_sa.l.w $ac2, $4, $3     # CHECK: dpaq_sa.l.w $ac2, $4, $3   # encoding: [0x00,0x64,0x92,0xbc]
  dpau.h.qbl $ac1, $3, $4      # CHECK: dpau.h.qbl $ac1, $3, $4    # encoding: [0x00,0x83,0x60,0xbc]
  dpau.h.qbr $ac2, $20, $21    # CHECK: dpau.h.qbr $ac2, $20, $21  # encoding: [0x02,0xb4,0xb0,0xbc]
