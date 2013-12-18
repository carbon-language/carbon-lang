# RUN: llvm-mc %s -arch=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# RUN: llvm-mc %s -arch=mips -mcpu=mips32r2 -mattr=+msa -filetype=obj -o - | \
# RUN: llvm-objdump -d -arch=mips -mattr=+msa - | \
# RUN: FileCheck %s -check-prefix=CHECKOBJDUMP
#

# CHECK:        ldi.b   $w8, 198                # encoding: [0x7b,0x06,0x32,0x07]
# CHECK:        ldi.h   $w20, 313               # encoding: [0x7b,0x29,0xcd,0x07]
# CHECK:        ldi.w   $w24, 492               # encoding: [0x7b,0x4f,0x66,0x07]
# CHECK:        ldi.d   $w27, -180              # encoding: [0x7b,0x7a,0x66,0xc7]

# CHECKOBJDUMP:        ldi.b   $w8, 198
# CHECKOBJDUMP:        ldi.h   $w20, 313
# CHECKOBJDUMP:        ldi.w   $w24, 492
# CHECKOBJDUMP:        ldi.d   $w27, 844

                ldi.b   $w8, 198
                ldi.h   $w20, 313
                ldi.w   $w24, 492
                ldi.d   $w27, -180
