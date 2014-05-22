# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r6 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

        .set noat
        bovc     $0, $2, 4       # TODO: bovc $0, $2, 4      # encoding: [0x20,0x40,0x00,0x01]
        bovc     $2, $4, 4       # TODO: bovc $2, $4, 4      # encoding: [0x20,0x82,0x00,0x01]
        bnvc     $0, $2, 4       # TODO: bnvc $0, $2, 4      # encoding: [0x60,0x40,0x00,0x01]
        bnvc     $2, $4, 4       # TODO: bnvc $2, $4, 4      # encoding: [0x60,0x82,0x00,0x01]
        beqc    $0, $6, 256      # TODO: beqc $6, $zero, 256 # encoding: [0x20,0xc0,0x00,0x40]
        beqc    $5, $0, 256      # TODO: beqc $5, $zero, 256 # encoding: [0x20,0xa0,0x00,0x40]
        beqc    $6, $5, 256      # TODO: beqc $5, $6, 256    # encoding: [0x20,0xa6,0x00,0x40]
        bnec    $0, $6, 256      # TODO: bnec $6, $zero, 256 # encoding: [0x60,0xc0,0x00,0x40]
        bnec    $5, $0, 256      # TODO: bnec $5, $zero, 256 # encoding: [0x60,0xa0,0x00,0x40]
        bnec    $6, $5, 256      # TODO: bnec $5, $6, 256    # encoding: [0x60,0xa6,0x00,0x40]
