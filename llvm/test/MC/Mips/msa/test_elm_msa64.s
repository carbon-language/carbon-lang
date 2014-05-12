# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        copy_s.d        $19, $w31[0]             # encoding: [0x78,0xb8,0xfc,0xd9]
# CHECK:        copy_u.d        $18, $w29[1]             # encoding: [0x78,0xf9,0xec,0x99]

        copy_s.d        $19, $w31[0]
        copy_u.d        $18, $w29[1]
