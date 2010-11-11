# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to make sure that all of the TYPE-B instructions supported by
# the Microblaze can be parsed by the assembly parser.

# TYPE B:   OPCODE RD    RA    IMMEDIATE
#           000000 00000 00000 0000000000000000

# CHECK:    addi
# BINARY:   001000 00001 00010 0000000000001111
# CHECK:    encoding: [0x20,0x22,0x00,0x0f]
            addi    r1, r2, 0x000F

# CHECK:    addic
# BINARY:   001010 00001 00010 0000000000001111
# CHECK:    encoding: [0x28,0x22,0x00,0x0f]
            addic   r1, r2, 0x000F

# CHECK:    addik
# BINARY:   001100 00001 00010 0000000000001111
# CHECK:    encoding: [0x30,0x22,0x00,0x0f]
            addik   r1, r2, 0x000F

# CHECK:    addikc
# BINARY:   001110 00001 00010 0000000000001111
# CHECK:    encoding: [0x38,0x22,0x00,0x0f]
            addikc  r1, r2, 0x000F

# CHECK:    andi
# BINARY:   101001 00001 00010 0000000000001111
# CHECK:    encoding: [0xa4,0x22,0x00,0x0f]
            andi    r1, r2, 0x000F

# CHECK:    andni
# BINARY:   101011 00001 00010 0000000000001111
# CHECK:    encoding: [0xac,0x22,0x00,0x0f]
            andni   r1, r2, 0x000F
