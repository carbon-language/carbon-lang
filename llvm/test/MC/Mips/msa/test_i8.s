# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 -mattr=+msa -arch=mips | FileCheck %s
#
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -mattr=+msa -arch=mips -filetype=obj -o - | llvm-objdump -d -triple=mipsel-unknown-linux -mattr=+msa -arch=mips - | FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        andi.b  $w2, $w29, 48           # encoding: [0x78,0x30,0xe8,0x80]
# CHECK:        bmnzi.b $w6, $w22, 126          # encoding: [0x78,0x7e,0xb1,0x81]
# CHECK:        bmzi.b  $w27, $w1, 88           # encoding: [0x79,0x58,0x0e,0xc1]
# CHECK:        bseli.b $w29, $w3, 189          # encoding: [0x7a,0xbd,0x1f,0x41]
# CHECK:        nori.b  $w1, $w17, 56           # encoding: [0x7a,0x38,0x88,0x40]
# CHECK:        ori.b   $w26, $w20, 135         # encoding: [0x79,0x87,0xa6,0x80]
# CHECK:        shf.b   $w19, $w30, 105         # encoding: [0x78,0x69,0xf4,0xc2]
# CHECK:        shf.h   $w17, $w8, 76           # encoding: [0x79,0x4c,0x44,0x42]
# CHECK:        shf.w   $w14, $w3, 93           # encoding: [0x7a,0x5d,0x1b,0x82]
# CHECK:        xori.b  $w16, $w10, 20          # encoding: [0x7b,0x14,0x54,0x00]

# CHECKOBJDUMP:        andi.b  $w2, $w29, 48
# CHECKOBJDUMP:        bmnzi.b $w6, $w22, 126
# CHECKOBJDUMP:        bmzi.b  $w27, $w1, 88
# CHECKOBJDUMP:        bseli.b $w29, $w3, 189
# CHECKOBJDUMP:        nori.b  $w1, $w17, 56
# CHECKOBJDUMP:        ori.b   $w26, $w20, 135
# CHECKOBJDUMP:        shf.b   $w19, $w30, 105
# CHECKOBJDUMP:        shf.h   $w17, $w8, 76
# CHECKOBJDUMP:        shf.w   $w14, $w3, 93
# CHECKOBJDUMP:        xori.b  $w16, $w10, 20

                andi.b  $w2, $w29, 48
                bmnzi.b $w6, $w22, 126
                bmzi.b  $w27, $w1, 88
                bseli.b $w29, $w3, 189
                nori.b  $w1, $w17, 56
                ori.b   $w26, $w20, 135
                shf.b   $w19, $w30, 105
                shf.h   $w17, $w8, 76
                shf.w   $w14, $w3, 93
                xori.b  $w16, $w10, 20
