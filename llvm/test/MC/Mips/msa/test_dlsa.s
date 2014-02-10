# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -mattr=+msa -show-encoding | \
# RUN:   FileCheck %s
#
# RUN: llvm-mc %s -arch=mips -mcpu=mips64r2 -mattr=+msa -filetype=obj -o - | \
# RUN:   llvm-objdump -d -arch=mips64 -mattr=+msa - | \
# RUN:     FileCheck %s -check-prefix=CHECKOBJDUMP
#
# CHECK:        dlsa        $8, $9, $10, 1              # encoding: [0x01,0x2a,0x40,0x15]
# CHECK:        dlsa        $8, $9, $10, 2              # encoding: [0x01,0x2a,0x40,0x55]
# CHECK:        dlsa        $8, $9, $10, 3              # encoding: [0x01,0x2a,0x40,0x95]
# CHECK:        dlsa        $8, $9, $10, 4              # encoding: [0x01,0x2a,0x40,0xd5]

# CHECKOBJDUMP: dlsa        $8, $9, $10, 1
# CHECKOBJDUMP: dlsa        $8, $9, $10, 2
# CHECKOBJDUMP: dlsa        $8, $9, $10, 3
# CHECKOBJDUMP: dlsa        $8, $9, $10, 4

                dlsa        $8, $9, $10, 1
                dlsa        $8, $9, $10, 2
                dlsa        $8, $9, $10, 3
                dlsa        $8, $9, $10, 4
