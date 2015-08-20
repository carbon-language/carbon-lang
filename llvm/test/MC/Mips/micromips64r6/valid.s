# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64r6 -mattr=micromips | FileCheck %s
a:
        .set noat
        addiur1sp $7, 4          # CHECK: addiur1sp $7, 4     # encoding: [0x6f,0x83]
        addiur2 $6, $7, -1       # CHECK: addiur2 $6, $7, -1  # encoding: [0x6f,0x7e]
        addiur2 $6, $7, 12       # CHECK: addiur2 $6, $7, 12  # encoding: [0x6f,0x76]
        addius5 $7, -2           # CHECK: addius5 $7, -2      # encoding: [0x4c,0xfc]
        addiusp -1028            # CHECK: addiusp -1028       # encoding: [0x4f,0xff]
        addiusp -1032            # CHECK: addiusp -1032       # encoding: [0x4f,0xfd]
        addiusp 1024             # CHECK: addiusp 1024        # encoding: [0x4c,0x01]
        addiusp 1028             # CHECK: addiusp 1028        # encoding: [0x4c,0x03]
        addiusp -16              # CHECK: addiusp -16         # encoding: [0x4f,0xf9]
        daui $3, $4, 5           # CHECK: daui $3, $4, 5      # encoding: [0xf0,0x64,0x00,0x05]
        dahi $3, 4               # CHECK: dahi $3, 4          # encoding: [0x42,0x23,0x00,0x04]
        dati $3, 4               # CHECK: dati $3, 4          # encoding: [0x42,0x03,0x00,0x04]
        dext $9, $6, 3, 7        # CHECK: dext $9, $6, 3, 7   # encoding: [0x59,0x26,0x30,0xec]
        dextm $9, $6, 3, 7       # CHECK: dextm $9, $6, 3, 7  # encoding: [0x59,0x26,0x30,0xe4]
        dextu $9, $6, 3, 7       # CHECK: dextu $9, $6, 3, 7  # encoding: [0x59,0x26,0x30,0xd4]
        dalign $4, $2, $3, 5     # CHECK: dalign $4, $2, $3, 5  # encoding: [0x58,0x43,0x25,0x1c]
        ddiv $3, $4, $5          # CHECK: ddiv $3, $4, $5     # encoding: [0x58,0x64,0x29,0x18]
        dmod $3, $4, $5          # CHECK: dmod $3, $4, $5     # encoding: [0x58,0x64,0x29,0x58]
        ddivu $3, $4, $5         # CHECK: ddivu $3, $4, $5    # encoding: [0x58,0x64,0x29,0x98]
        dmodu $3, $4, $5         # CHECK: dmodu $3, $4, $5    # encoding: [0x58,0x64,0x29,0xd8]

1:
