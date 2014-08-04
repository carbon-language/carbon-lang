# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Instructions specific to the PowerPC 4xx embedded controllers:

# CHECK-BE: mfdcr 3, 178                     # encoding: [0x7c,0x72,0x2a,0x86]
# CHECK-LE: mfdcr 3, 178                     # encoding: [0x86,0x2a,0x72,0x7c]
            mfdcr 3,178
# CHECK-BE: mtdcr 178, 3                     # encoding: [0x7c,0x72,0x2b,0x86]
# CHECK-LE: mtdcr 178, 3                     # encoding: [0x86,0x2b,0x72,0x7c]
            mtdcr 178,3

# CHECK-BE: tlbre 2, 3, 0                    # encoding: [0x7c,0x43,0x07,0x64]
# CHECK-LE: tlbre 2, 3, 0                    # encoding: [0x64,0x07,0x43,0x7c]
            tlbre %r2, %r3, 0
# CHECK-BE: tlbre 2, 3, 1                    # encoding: [0x7c,0x43,0x0f,0x64]
# CHECK-LE: tlbre 2, 3, 1                    # encoding: [0x64,0x0f,0x43,0x7c]
            tlbre %r2, %r3, 1
# CHECK-BE: tlbre 2, 3, 0                    # encoding: [0x7c,0x43,0x07,0x64]
# CHECK-LE: tlbre 2, 3, 0                    # encoding: [0x64,0x07,0x43,0x7c]
            tlbrehi %r2, %r3
# CHECK-BE: tlbre 2, 3, 1                    # encoding: [0x7c,0x43,0x0f,0x64]
# CHECK-LE: tlbre 2, 3, 1                    # encoding: [0x64,0x0f,0x43,0x7c]
            tlbrelo %r2, %r3

# CHECK-BE: tlbwe 2, 3, 0                    # encoding: [0x7c,0x43,0x07,0xa4]
# CHECK-LE: tlbwe 2, 3, 0                    # encoding: [0xa4,0x07,0x43,0x7c]
            tlbwe %r2, %r3, 0
# CHECK-BE: tlbwe 2, 3, 1                    # encoding: [0x7c,0x43,0x0f,0xa4]
# CHECK-LE: tlbwe 2, 3, 1                    # encoding: [0xa4,0x0f,0x43,0x7c]
            tlbwe %r2, %r3, 1
# CHECK-BE: tlbwe 2, 3, 0                    # encoding: [0x7c,0x43,0x07,0xa4]
# CHECK-LE: tlbwe 2, 3, 0                    # encoding: [0xa4,0x07,0x43,0x7c]
            tlbwehi %r2, %r3
# CHECK-BE: tlbwe 2, 3, 1                    # encoding: [0x7c,0x43,0x0f,0xa4]
# CHECK-LE: tlbwe 2, 3, 1                    # encoding: [0xa4,0x0f,0x43,0x7c]
            tlbwelo %r2, %r3

# CHECK-BE: tlbsx 2, 3, 1                    # encoding: [0x7c,0x43,0x0f,0x24]
# CHECK-LE: tlbsx 2, 3, 1                    # encoding: [0x24,0x0f,0x43,0x7c]
            tlbsx %r2, %r3, %r1
# CHECK-BE: tlbsx. 2, 3, 1                   # encoding: [0x7c,0x43,0x0f,0x25]
# CHECK-LE: tlbsx. 2, 3, 1                   # encoding: [0x25,0x0f,0x43,0x7c]
            tlbsx. %r2, %r3, %r1
