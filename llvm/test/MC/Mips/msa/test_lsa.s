# RUN: llvm-mc %s -arch=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# RUN: llvm-mc %s -arch=mips -mcpu=mips32r2 -mattr=+msa -filetype=obj -o - | \
# RUN: llvm-objdump -d -arch=mips -mattr=+msa - | \
# RUN: FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        lsa        $8, $9, $10, 1              # encoding: [0x01,0x2a,0x40,0x05]
# CHECK:        lsa        $8, $9, $10, 2              # encoding: [0x01,0x2a,0x40,0x45]
# CHECK:        lsa        $8, $9, $10, 3              # encoding: [0x01,0x2a,0x40,0x85]
# CHECK:        lsa        $8, $9, $10, 4              # encoding: [0x01,0x2a,0x40,0xc5]

# CHECKOBJDUMP: lsa        $8, $9, $10, 1
# CHECKOBJDUMP: lsa        $8, $9, $10, 2
# CHECKOBJDUMP: lsa        $8, $9, $10, 3
# CHECKOBJDUMP: lsa        $8, $9, $10, 4

                lsa        $8, $9, $10, 1
                lsa        $8, $9, $10, 2
                lsa        $8, $9, $10, 3
                lsa        $8, $9, $10, 4
