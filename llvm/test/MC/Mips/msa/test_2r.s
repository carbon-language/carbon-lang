# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 -mattr=+msa -arch=mips | FileCheck %s
#
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -mattr=+msa -arch=mips -filetype=obj -o - | llvm-objdump -d -triple=mipsel-unknown-linux -mattr=+msa -arch=mips - | FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        fill.b  $w30, $9                # encoding: [0x7b,0x00,0x4f,0x9e]
# CHECK:        fill.h  $w31, $23               # encoding: [0x7b,0x01,0xbf,0xde]
# CHECK:        fill.w  $w16, $24               # encoding: [0x7b,0x02,0xc4,0x1e]

# CHECKOBJDUMP:        fill.b  $w30, $9
# CHECKOBJDUMP:        fill.h  $w31, $23
# CHECKOBJDUMP:        fill.w  $w16, $24

                fill.b  $w30, $9
                fill.h  $w31, $23
                fill.w  $w16, $24
