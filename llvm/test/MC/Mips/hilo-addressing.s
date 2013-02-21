# RUN: llvm-mc -show-encoding -triple mips-unknown-unknown %s | FileCheck %s

  .ent hilo_test
     .equ    addr, 0xdeadbeef
# CHECK: # encoding: [0x3c,0x04,0xde,0xae]
    lui $4,%hi(addr)
# CHECK: # encoding: [0x03,0xe0,0x00,0x08]
    jr  $31
# CHECK: # encoding: [0x80,0x82,0xbe,0xef]
    lb  $2,%lo(addr)($4)
    .end hilo_test
