# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 -mattr=+msa -arch=mips | FileCheck %s
#
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -mattr=+msa -arch=mips -filetype=obj -o - | llvm-objdump -d -triple=mipsel-unknown-linux -mattr=+msa -arch=mips - | FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        lsa        $8, $9, $10, 2              # encoding: [0x01,0x2a,0x40,0x85]

# CHECKOBJDUMP: lsa        $8, $9, $10, 2

                lsa        $8, $9, $10, 2
