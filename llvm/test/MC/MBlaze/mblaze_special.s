# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to ensure that all FPU instructions can be parsed by the
# assembly parser correctly.

# TYPE A:   OPCODE RD    RA    RB    FLAGS
# BINARY:   011011 00000 00000 00000 00000000000

# CHECK:    mfs
# BINARY:   100101 00000 00000 10000 00000000000
# CHECK:    encoding: [0x94,0x00,0x80,0x00]
            mfs         r0, 0x0

# CHECK:    msrclr
# BINARY:   100101 00000 100010 000000000000000
# CHECK:    encoding: [0x94,0x11,0x00,0x00]
            msrclr      r0, 0x0

# CHECK:    msrset
# BINARY:   100101 00000 100000 000000000000000
# CHECK:    encoding: [0x94,0x10,0x00,0x00]
            msrset      r0, 0x0

# CHECK:    mts
# BINARY:   100101 00000 00000 11 00000000000000
# CHECK:    encoding: [0x94,0x00,0xc0,0x00]
            mts         0x0 , r0

# CHECK:    wdc
# BINARY:   100100 00000 00000 00001 00001100100
# CHECK:    encoding: [0x90,0x00,0x08,0x64]
            wdc         r0, r1

# CHECK:    wdc.clear
# BINARY:   100100 00000 00000 00001 00001100110
# CHECK:    encoding: [0x90,0x00,0x08,0x66]
            wdc.clear   r0, r1

# CHECK:    wdc.flush
# BINARY:   100100 00000 00000 00001 00001110100
# CHECK:    encoding: [0x90,0x00,0x08,0x74]
            wdc.flush   r0, r1

# CHECK:    wic
# BINARY:   100100 00000 00000 00001 00001101000
# CHECK:    encoding: [0x90,0x00,0x08,0x68]
            wic         r0, r1
