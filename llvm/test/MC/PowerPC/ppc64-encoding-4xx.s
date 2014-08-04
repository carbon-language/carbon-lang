# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Instructions specific to the PowerPC 4xx embedded controllers:

# CHECK-BE: mfdcr 3, 178                     # encoding: [0x7c,0x72,0x2a,0x86]
# CHECK-LE: mfdcr 3, 178                     # encoding: [0x86,0x2a,0x72,0x7c]
            mfdcr 3,178
# CHECK-BE: mtdcr 178, 3                     # encoding: [0x7c,0x72,0x2b,0x86]
# CHECK-LE: mtdcr 178, 3                     # encoding: [0x86,0x2b,0x72,0x7c]
            mtdcr 178,3
