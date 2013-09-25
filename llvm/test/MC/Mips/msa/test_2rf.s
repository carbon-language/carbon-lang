# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 -mattr=+msa -arch=mips | FileCheck %s
#
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -mattr=+msa -arch=mips -filetype=obj -o - | llvm-objdump -d -triple=mipsel-unknown-linux -mattr=+msa -arch=mips - | FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        fclass.w        $w26, $w12              # encoding: [0x7b,0x20,0x66,0x9e]
# CHECK:        fclass.d        $w24, $w17              # encoding: [0x7b,0x21,0x8e,0x1e]

# CHECKOBJDUMP:        fclass.w        $w26, $w12
# CHECKOBJDUMP:        fclass.d        $w24, $w17

                fclass.w        $w26, $w12
                fclass.d        $w24, $w17
