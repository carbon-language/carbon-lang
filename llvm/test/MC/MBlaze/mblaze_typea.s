# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to make sure that all of the TYPE-A instructions supported by
# the Microblaze can be parsed by the assembly parser.

# TYPE A:   OPCODE RD    RA    RB    FLAGS
# BINARY:   000000 00000 00000 00000 00000000000

# CHECK:    add
# BINARY:   000000 00001 00010 00011 00000000000
# CHECK:    encoding: [0x00,0x22,0x18,0x00]
            add     r1, r2, r3

# CHECK:    addc
# BINARY:   000010 00001 00010 00011 00000000000
# CHECK:    encoding: [0x08,0x22,0x18,0x00]
            addc    r1, r2, r3

# CHECK:    addk
# BINARY:   000100 00001 00010 00011 00000000000
# CHECK:    encoding: [0x10,0x22,0x18,0x00]
            addk    r1, r2, r3

# CHECK:    addkc
# BINARY:   000110 00001 00010 00011 00000000000
# CHECK:    encoding: [0x18,0x22,0x18,0x00]
            addkc   r1, r2, r3

# CHECK:    and
# BINARY:   100001 00001 00010 00011 00000000000
# CHECK:    encoding: [0x84,0x22,0x18,0x00]
            and     r1, r2, r3

# CHECK:    andn
# BINARY:   100011 00001 00010 00011 00000000000
# CHECK:    encoding: [0x8c,0x22,0x18,0x00]
            andn    r1, r2, r3

# CHECK:    beq
# BINARY:   100111 00000 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0x02,0x18,0x00]
            beq     r2, r3

# CHECK:    bge
# BINARY:   100111 00101 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0xa2,0x18,0x00]
            bge     r2, r3

# CHECK:    bgt
# BINARY:   100111 00100 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0x82,0x18,0x00]
            bgt     r2, r3

# CHECK:    ble
# BINARY:   100111 00011 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0x62,0x18,0x00]
            ble     r2, r3

# CHECK:    blt
# BINARY:   100111 00010 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0x42,0x18,0x00]
            blt     r2, r3

# CHECK:    bne
# BINARY:   100111 00001 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0x22,0x18,0x00]
            bne     r2, r3

# CHECK:    nop
# BINARY:   100000 00000 00000 00000 00000000000
# CHECK:    encoding: [0x80,0x00,0x00,0x00]
        nop
