# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -mattr=+msa -show-encoding | FileCheck %s
#
# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -mattr=+msa -filetype=obj -o - | \
# RUN:   llvm-objdump -d -arch=mips64 -mattr=+msa - | \
# RUN:     FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        insert.d        $w1[1], $sp            # encoding: [0x79,0x39,0xe8,0x59]

# CHECKOBJDUMP:        insert.d        $w1[1], $sp

                insert.d        $w1[1], $sp
