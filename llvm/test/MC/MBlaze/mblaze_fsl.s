# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to ensure that all FSL immediate operands and FSL instructions
# can be parsed by the assembly parser correctly.

# TYPE F:   OPCODE RD           NCTAE        FSL
# BINARY:   011011 00000 000000 00000 000000 0000

# TYPE FD:  OPCODE RD          RB      NCTAE
# BINARY:   011011 00000 00000 00000 0 00000 00000

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x00,0x00]
            get         r0, rfsl0

# CHECK:    nget
# BINARY:   011011 00000 000000 10000 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x40,0x00]
            nget        r0, rfsl0

# CHECK:    cget
# BINARY:   011011 00000 000000 01000 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x20,0x00]
            cget        r0, rfsl0

# CHECK:    ncget
# BINARY:   011011 00000 000000 11000 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x60,0x00]
            ncget       r0, rfsl0

# CHECK:    tget
# BINARY:   011011 00000 000000 00100 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x10,0x00]
            tget        r0, rfsl0

# CHECK:    tnget
# BINARY:   011011 00000 000000 10100 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x50,0x00]
            tnget       r0, rfsl0

# CHECK:    tcget
# BINARY:   011011 00000 000000 01100 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x30,0x00]
            tcget       r0, rfsl0

# CHECK:    tncget
# BINARY:   011011 00000 000000 11100 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x70,0x00]
            tncget      r0, rfsl0

# CHECK:    aget
# BINARY:   011011 00000 000000 00010 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x08,0x00]
            aget        r0, rfsl0

# CHECK:    naget
# BINARY:   011011 00000 000000 10010 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x48,0x00]
            naget       r0, rfsl0

# CHECK:    caget
# BINARY:   011011 00000 000000 01010 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x28,0x00]
            caget       r0, rfsl0

# CHECK:    ncaget
# BINARY:   011011 00000 000000 11010 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x68,0x00]
            ncaget      r0, rfsl0

# CHECK:    taget
# BINARY:   011011 00000 000000 00110 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x18,0x00]
            taget       r0, rfsl0

# CHECK:    tnaget
# BINARY:   011011 00000 000000 10110 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x58,0x00]
            tnaget      r0, rfsl0

# CHECK:    tcaget
# BINARY:   011011 00000 000000 01110 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x38,0x00]
            tcaget      r0, rfsl0

# CHECK:    tncaget
# BINARY:   011011 00000 000000 11110 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x78,0x00]
            tncaget     r0, rfsl0

# CHECK:    eget
# BINARY:   011011 00000 000000 00001 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x04,0x00]
            eget        r0, rfsl0

# CHECK:    neget
# BINARY:   011011 00000 000000 10001 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x44,0x00]
            neget       r0, rfsl0

# CHECK:    ecget
# BINARY:   011011 00000 000000 01001 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x24,0x00]
            ecget       r0, rfsl0

# CHECK:    necget
# BINARY:   011011 00000 000000 11001 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x64,0x00]
            necget      r0, rfsl0

# CHECK:    teget
# BINARY:   011011 00000 000000 00101 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x14,0x00]
            teget       r0, rfsl0

# CHECK:    tneget
# BINARY:   011011 00000 000000 10101 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x54,0x00]
            tneget      r0, rfsl0

# CHECK:    tecget
# BINARY:   011011 00000 000000 01101 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x34,0x00]
            tecget      r0, rfsl0

# CHECK:    tnecget
# BINARY:   011011 00000 000000 11101 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x74,0x00]
            tnecget     r0, rfsl0

# CHECK:    eaget
# BINARY:   011011 00000 000000 00011 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x0c,0x00]
            eaget       r0, rfsl0

# CHECK:    neaget
# BINARY:   011011 00000 000000 10011 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x4c,0x00]
            neaget      r0, rfsl0

# CHECK:    ecaget
# BINARY:   011011 00000 000000 01011 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x2c,0x00]
            ecaget      r0, rfsl0

# CHECK:    necaget
# BINARY:   011011 00000 000000 11011 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x6c,0x00]
            necaget     r0, rfsl0

# CHECK:    teaget
# BINARY:   011011 00000 000000 00111 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x1c,0x00]
            teaget      r0, rfsl0

# CHECK:    tneaget
# BINARY:   011011 00000 000000 10111 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x5c,0x00]
            tneaget     r0, rfsl0

# CHECK:    tecaget
# BINARY:   011011 00000 000000 01111 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x3c,0x00]
            tecaget     r0, rfsl0

# CHECK:    tnecaget
# BINARY:   011011 00000 000000 11111 000000 0000
# CHECK:    encoding: [0x6c,0x00,0x7c,0x00]
            tnecaget    r0, rfsl0

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0001
# CHECK:    encoding: [0x6c,0x00,0x00,0x01]
            get     r0, rfsl1

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0010
# CHECK:    encoding: [0x6c,0x00,0x00,0x02]
            get     r0, rfsl2

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0011
# CHECK:    encoding: [0x6c,0x00,0x00,0x03]
            get     r0, rfsl3

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0100
# CHECK:    encoding: [0x6c,0x00,0x00,0x04]
            get     r0, rfsl4

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0101
# CHECK:    encoding: [0x6c,0x00,0x00,0x05]
            get     r0, rfsl5

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0110
# CHECK:    encoding: [0x6c,0x00,0x00,0x06]
            get     r0, rfsl6

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 0111
# CHECK:    encoding: [0x6c,0x00,0x00,0x07]
            get     r0, rfsl7

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1000
# CHECK:    encoding: [0x6c,0x00,0x00,0x08]
            get     r0, rfsl8

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1001
# CHECK:    encoding: [0x6c,0x00,0x00,0x09]
            get     r0, rfsl9

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1010
# CHECK:    encoding: [0x6c,0x00,0x00,0x0a]
            get     r0, rfsl10

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1011
# CHECK:    encoding: [0x6c,0x00,0x00,0x0b]
            get     r0, rfsl11

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1100
# CHECK:    encoding: [0x6c,0x00,0x00,0x0c]
            get     r0, rfsl12

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1101
# CHECK:    encoding: [0x6c,0x00,0x00,0x0d]
            get     r0, rfsl13

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1110
# CHECK:    encoding: [0x6c,0x00,0x00,0x0e]
            get     r0, rfsl14

# CHECK:    get
# BINARY:   011011 00000 000000 00000 000000 1111
# CHECK:    encoding: [0x6c,0x00,0x00,0x0f]
            get     r0, rfsl15
