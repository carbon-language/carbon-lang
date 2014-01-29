# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -mattr=+msa -show-encoding | FileCheck %s
#
# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -mattr=+msa -filetype=obj -o - | \
# RUN:   llvm-objdump -d -arch=mips64 -mattr=+msa - | \
# RUN:     FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        copy_s.d        $19, $w31[0]             # encoding: [0x78,0xb8,0xfc,0xd9]
# CHECK:        copy_u.d        $18, $w29[1]             # encoding: [0x78,0xf9,0xec,0x99]

# CHECKOBJDUMP:        copy_s.d        $19, $w31[0]
# CHECKOBJDUMP:        copy_u.d        $18, $w29[1]

        copy_s.d        $19, $w31[0]
        copy_u.d        $18, $w29[1]
