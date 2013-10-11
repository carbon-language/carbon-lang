# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 -mattr=+msa -arch=mips | FileCheck %s
#
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -mattr=+msa -arch=mips -filetype=obj -o - | llvm-objdump -d -triple=mipsel-unknown-linux -mattr=+msa -arch=mips - | FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        insert.b        $w23[3], $sp            # encoding: [0x79,0x03,0xed,0xd9]
# CHECK:        insert.h        $w20[2], $5             # encoding: [0x79,0x22,0x2d,0x19]
# CHECK:        insert.w        $w8[2], $15             # encoding: [0x79,0x32,0x7a,0x19]

# CHECKOBJDUMP:        insert.b        $w23[3], $sp
# CHECKOBJDUMP:        insert.h        $w20[2], $5
# CHECKOBJDUMP:        insert.w        $w8[2], $15

                insert.b        $w23[3], $sp
                insert.h        $w20[2], $5
                insert.w        $w8[2], $15
