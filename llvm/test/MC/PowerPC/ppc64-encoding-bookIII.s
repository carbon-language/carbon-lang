# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# CHECK: mtmsr 4, 0                       # encoding: [0x7c,0x80,0x01,0x24]
         mtmsr %r4

# CHECK: mtmsr 4, 1                       # encoding: [0x7c,0x81,0x01,0x24]
         mtmsr %r4, 1

# CHECK: mfmsr 4                         # encoding: [0x7c,0x80,0x00,0xa6]
         mfmsr %r4

# CHECK: mtmsrd 4, 0                     # encoding: [0x7c,0x80,0x01,0x64]
         mtmsrd %r4

# CHECK: mtmsrd 4, 1                     # encoding: [0x7c,0x81,0x01,0x64]
         mtmsrd %r4, 1

# CHECK: mfspr 4, 272                    # encoding: [0x7c,0x90,0x42,0xa6]
         mfsprg %r4, 0

# CHECK: mfspr 4, 273                    # encoding: [0x7c,0x91,0x42,0xa6]
         mfsprg %r4, 1

# CHECK: mfspr 4, 274                    # encoding: [0x7c,0x92,0x42,0xa6]
         mfsprg %r4, 2

# CHECK: mfspr 4, 275                    # encoding: [0x7c,0x93,0x42,0xa6]
         mfsprg %r4, 3

# CHECK: mtspr 272, 4                    # encoding: [0x7c,0x90,0x43,0xa6]
         mtsprg 0, %r4

# CHECK: mtspr 273, 4                    # encoding: [0x7c,0x91,0x43,0xa6]
         mtsprg 1, %r4

# CHECK: mtspr 274, 4                    # encoding: [0x7c,0x92,0x43,0xa6]
         mtsprg 2, %r4

# CHECK: mtspr 275, 4                    # encoding: [0x7c,0x93,0x43,0xa6]
         mtsprg 3, %r4

# CHECK: mtspr 272, 4                    # encoding: [0x7c,0x90,0x43,0xa6]
         mtsprg0 %r4

# CHECK: mtspr 273, 4                    # encoding: [0x7c,0x91,0x43,0xa6]
         mtsprg1 %r4

# CHECK: mtspr 274, 4                    # encoding: [0x7c,0x92,0x43,0xa6]
         mtsprg2 %r4

# CHECK: mtspr 275, 4                    # encoding: [0x7c,0x93,0x43,0xa6]
         mtsprg3 %r4

# CHECK: mtspr 280, 4                    # encoding: [0x7c,0x98,0x43,0xa6]
         mtasr %r4

# CHECK: mfspr 4, 22                     # encoding: [0x7c,0x96,0x02,0xa6]
         mfdec %r4

# CHECK: mtspr 22, 4                     # encoding: [0x7c,0x96,0x03,0xa6]
         mtdec %r4

# CHECK: mfspr 4, 287                    # encoding: [0x7c,0x9f,0x42,0xa6]
         mfpvr %r4

# CHECK: mfspr 4, 25                     # encoding: [0x7c,0x99,0x02,0xa6]
         mfsdr1 %r4

# CHECK: mtspr 25, 4                     # encoding: [0x7c,0x99,0x03,0xa6]
         mtsdr1 %r4

# CHECK: mfspr 4, 26                     # encoding: [0x7c,0x9a,0x02,0xa6]
         mfsrr0 %r4

# CHECK: mtspr 26, 4                     # encoding: [0x7c,0x9a,0x03,0xa6]
         mtsrr0 %r4

# CHECK: mfspr 4, 27                     # encoding: [0x7c,0x9b,0x02,0xa6]
         mfsrr1 %r4

# CHECK: mtspr 27, 4                     # encoding: [0x7c,0x9b,0x03,0xa6]
         mtsrr1 %r4

# CHECK: slbie 4                         # encoding: [0x7c,0x00,0x23,0x64]
         slbie %r4

# CHECK: slbmte 4, 5                     # encoding: [0x7c,0x80,0x2b,0x24]
         slbmte %r4, %r5

# CHECK: slbmfee 4, 5                    # encoding: [0x7c,0x80,0x2f,0x26]
         slbmfee %r4, %r5

# CHECK: slbia                           # encoding: [0x7c,0x00,0x03,0xe4]
         slbia

# CHECK: tlbsync                         # encoding: [0x7c,0x00,0x04,0x6c]
         tlbsync

# CHECK: tlbiel 4                        # encoding: [0x7c,0x00,0x22,0x24]
         tlbiel %r4

# CHECK: tlbie 4,0                       # encoding: [0x7c,0x00,0x22,0x64]
         tlbie %r4, 0

# CHECK: tlbie 4,0                       # encoding: [0x7c,0x00,0x22,0x64]
         tlbie %r4

