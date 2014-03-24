# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# CHECK-BE: mtmsr 4, 0                       # encoding: [0x7c,0x80,0x01,0x24]
# CHECK-LE: mtmsr 4, 0                       # encoding: [0x24,0x01,0x80,0x7c]
            mtmsr %r4

# CHECK-BE: mtmsr 4, 1                       # encoding: [0x7c,0x81,0x01,0x24]
# CHECK-LE: mtmsr 4, 1                       # encoding: [0x24,0x01,0x81,0x7c]
            mtmsr %r4, 1

# CHECK-BE: mfmsr 4                         # encoding: [0x7c,0x80,0x00,0xa6]
# CHECK-LE: mfmsr 4                         # encoding: [0xa6,0x00,0x80,0x7c]
            mfmsr %r4

# CHECK-BE: mtmsrd 4, 0                     # encoding: [0x7c,0x80,0x01,0x64]
# CHECK-LE: mtmsrd 4, 0                     # encoding: [0x64,0x01,0x80,0x7c]
            mtmsrd %r4

# CHECK-BE: mtmsrd 4, 1                     # encoding: [0x7c,0x81,0x01,0x64]
# CHECK-LE: mtmsrd 4, 1                     # encoding: [0x64,0x01,0x81,0x7c]
            mtmsrd %r4, 1

# CHECK-BE: mfspr 4, 272                    # encoding: [0x7c,0x90,0x42,0xa6]
# CHECK-LE: mfspr 4, 272                    # encoding: [0xa6,0x42,0x90,0x7c]
            mfsprg %r4, 0

# CHECK-BE: mfspr 4, 273                    # encoding: [0x7c,0x91,0x42,0xa6]
# CHECK-LE: mfspr 4, 273                    # encoding: [0xa6,0x42,0x91,0x7c]
            mfsprg %r4, 1

# CHECK-BE: mfspr 4, 274                    # encoding: [0x7c,0x92,0x42,0xa6]
# CHECK-LE: mfspr 4, 274                    # encoding: [0xa6,0x42,0x92,0x7c]
            mfsprg %r4, 2

# CHECK-BE: mfspr 4, 275                    # encoding: [0x7c,0x93,0x42,0xa6]
# CHECK-LE: mfspr 4, 275                    # encoding: [0xa6,0x42,0x93,0x7c]
            mfsprg %r4, 3

# CHECK-BE: mtspr 272, 4                    # encoding: [0x7c,0x90,0x43,0xa6]
# CHECK-LE: mtspr 272, 4                    # encoding: [0xa6,0x43,0x90,0x7c]
            mtsprg 0, %r4

# CHECK-BE: mtspr 273, 4                    # encoding: [0x7c,0x91,0x43,0xa6]
# CHECK-LE: mtspr 273, 4                    # encoding: [0xa6,0x43,0x91,0x7c]
            mtsprg 1, %r4

# CHECK-BE: mtspr 274, 4                    # encoding: [0x7c,0x92,0x43,0xa6]
# CHECK-LE: mtspr 274, 4                    # encoding: [0xa6,0x43,0x92,0x7c]
            mtsprg 2, %r4

# CHECK-BE: mtspr 275, 4                    # encoding: [0x7c,0x93,0x43,0xa6]
# CHECK-LE: mtspr 275, 4                    # encoding: [0xa6,0x43,0x93,0x7c]
            mtsprg 3, %r4

# CHECK-BE: mtspr 272, 4                    # encoding: [0x7c,0x90,0x43,0xa6]
# CHECK-LE: mtspr 272, 4                    # encoding: [0xa6,0x43,0x90,0x7c]
            mtsprg0 %r4

# CHECK-BE: mtspr 273, 4                    # encoding: [0x7c,0x91,0x43,0xa6]
# CHECK-LE: mtspr 273, 4                    # encoding: [0xa6,0x43,0x91,0x7c]
            mtsprg1 %r4

# CHECK-BE: mtspr 274, 4                    # encoding: [0x7c,0x92,0x43,0xa6]
# CHECK-LE: mtspr 274, 4                    # encoding: [0xa6,0x43,0x92,0x7c]
            mtsprg2 %r4

# CHECK-BE: mtspr 275, 4                    # encoding: [0x7c,0x93,0x43,0xa6]
# CHECK-LE: mtspr 275, 4                    # encoding: [0xa6,0x43,0x93,0x7c]
            mtsprg3 %r4

# CHECK-BE: mtspr 280, 4                    # encoding: [0x7c,0x98,0x43,0xa6]
# CHECK-LE: mtspr 280, 4                    # encoding: [0xa6,0x43,0x98,0x7c]
            mtasr %r4

# CHECK-BE: mfspr 4, 22                     # encoding: [0x7c,0x96,0x02,0xa6]
# CHECK-LE: mfspr 4, 22                     # encoding: [0xa6,0x02,0x96,0x7c]
            mfdec %r4

# CHECK-BE: mtspr 22, 4                     # encoding: [0x7c,0x96,0x03,0xa6]
# CHECK-LE: mtspr 22, 4                     # encoding: [0xa6,0x03,0x96,0x7c]
            mtdec %r4

# CHECK-BE: mfspr 4, 287                    # encoding: [0x7c,0x9f,0x42,0xa6]
# CHECK-LE: mfspr 4, 287                    # encoding: [0xa6,0x42,0x9f,0x7c]
            mfpvr %r4

# CHECK-BE: mfspr 4, 25                     # encoding: [0x7c,0x99,0x02,0xa6]
# CHECK-LE: mfspr 4, 25                     # encoding: [0xa6,0x02,0x99,0x7c]
            mfsdr1 %r4

# CHECK-BE: mtspr 25, 4                     # encoding: [0x7c,0x99,0x03,0xa6]
# CHECK-LE: mtspr 25, 4                     # encoding: [0xa6,0x03,0x99,0x7c]
            mtsdr1 %r4

# CHECK-BE: mfspr 4, 26                     # encoding: [0x7c,0x9a,0x02,0xa6]
# CHECK-LE: mfspr 4, 26                     # encoding: [0xa6,0x02,0x9a,0x7c]
            mfsrr0 %r4

# CHECK-BE: mtspr 26, 4                     # encoding: [0x7c,0x9a,0x03,0xa6]
# CHECK-LE: mtspr 26, 4                     # encoding: [0xa6,0x03,0x9a,0x7c]
            mtsrr0 %r4

# CHECK-BE: mfspr 4, 27                     # encoding: [0x7c,0x9b,0x02,0xa6]
# CHECK-LE: mfspr 4, 27                     # encoding: [0xa6,0x02,0x9b,0x7c]
            mfsrr1 %r4

# CHECK-BE: mtspr 27, 4                     # encoding: [0x7c,0x9b,0x03,0xa6]
# CHECK-LE: mtspr 27, 4                     # encoding: [0xa6,0x03,0x9b,0x7c]
            mtsrr1 %r4

# CHECK-BE: slbie 4                         # encoding: [0x7c,0x00,0x23,0x64]
# CHECK-LE: slbie 4                         # encoding: [0x64,0x23,0x00,0x7c]
            slbie %r4

# CHECK-BE: slbmte 4, 5                     # encoding: [0x7c,0x80,0x2b,0x24]
# CHECK-LE: slbmte 4, 5                     # encoding: [0x24,0x2b,0x80,0x7c]
            slbmte %r4, %r5

# CHECK-BE: slbmfee 4, 5                    # encoding: [0x7c,0x80,0x2f,0x26]
# CHECK-LE: slbmfee 4, 5                    # encoding: [0x26,0x2f,0x80,0x7c]
            slbmfee %r4, %r5

# CHECK-BE: slbia                           # encoding: [0x7c,0x00,0x03,0xe4]
# CHECK-LE: slbia                           # encoding: [0xe4,0x03,0x00,0x7c]
            slbia

# CHECK-BE: tlbsync                         # encoding: [0x7c,0x00,0x04,0x6c]
# CHECK-LE: tlbsync                         # encoding: [0x6c,0x04,0x00,0x7c]
            tlbsync

# CHECK-BE: tlbiel 4                        # encoding: [0x7c,0x00,0x22,0x24]
# CHECK-LE: tlbiel 4                        # encoding: [0x24,0x22,0x00,0x7c]
            tlbiel %r4

# CHECK-BE: tlbie 4,0                       # encoding: [0x7c,0x00,0x22,0x64]
# CHECK-LE: tlbie 4,0                       # encoding: [0x64,0x22,0x00,0x7c]
            tlbie %r4, 0

# CHECK-BE: tlbie 4,0                       # encoding: [0x7c,0x00,0x22,0x64]
# CHECK-LE: tlbie 4,0                       # encoding: [0x64,0x22,0x00,0x7c]
            tlbie %r4

