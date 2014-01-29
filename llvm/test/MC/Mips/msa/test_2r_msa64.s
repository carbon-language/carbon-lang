# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -mattr=+msa -show-encoding | FileCheck %s
#
# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -mattr=+msa -filetype=obj -o - | \
# RUN:   llvm-objdump -d -arch=mips64 -mattr=+msa - | \
# RUN:     FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        fill.d  $w27, $9                # encoding: [0x7b,0x03,0x4e,0xde]

# CHECKOBJDUMP:        fill.d  $w27, $9

                fill.d  $w27, $9
