# Instructions that are valid
#
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64r6 | FileCheck %s

        .set noat
        # FIXME: Add the instructions carried forward from older ISA's
        div     $2,$3,$4         # CHECK: div $2, $3, $4   # encoding: [0x00,0x64,0x10,0x9a]
        divu    $2,$3,$4         # CHECK: divu $2, $3, $4  # encoding: [0x00,0x64,0x10,0x9b]
        mod     $2,$3,$4         # CHECK: mod $2, $3, $4   # encoding: [0x00,0x64,0x10,0xda]
        modu    $2,$3,$4         # CHECK: modu $2, $3, $4  # encoding: [0x00,0x64,0x10,0xdb]
        ddiv    $2,$3,$4         # CHECK: ddiv $2, $3, $4  # encoding: [0x00,0x64,0x10,0x9e]
        ddivu   $2,$3,$4         # CHECK: ddivu $2, $3, $4 # encoding: [0x00,0x64,0x10,0x9f]
        dmod    $2,$3,$4         # CHECK: dmod $2, $3, $4  # encoding: [0x00,0x64,0x10,0xde]
        dmodu   $2,$3,$4         # CHECK: dmodu $2, $3, $4 # encoding: [0x00,0x64,0x10,0xdf]
#        mul     $2,$3,$4         # CHECK-TODO: mul $2, $3, $4   # encoding: [0x00,0x64,0x10,0x98]
        muh     $2,$3,$4         # CHECK: muh $2, $3, $4   # encoding: [0x00,0x64,0x10,0xd8]
        mulu    $2,$3,$4         # CHECK: mulu $2, $3, $4  # encoding: [0x00,0x64,0x10,0x99]
        muhu    $2,$3,$4         # CHECK: muhu $2, $3, $4  # encoding: [0x00,0x64,0x10,0xd9]
        dmul    $2,$3,$4         # CHECK: dmul $2, $3, $4  # encoding: [0x00,0x64,0x10,0xb8]
        dmuh    $2,$3,$4         # CHECK: dmuh $2, $3, $4  # encoding: [0x00,0x64,0x10,0xf8]
        dmulu   $2,$3,$4         # CHECK: dmulu $2, $3, $4 # encoding: [0x00,0x64,0x10,0xb9]
        dmuhu   $2,$3,$4         # CHECK: dmuhu $2, $3, $4 # encoding: [0x00,0x64,0x10,0xf9]
