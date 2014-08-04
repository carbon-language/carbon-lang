# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Instructions specific to the PowerPC 6xx family:

# CHECK-BE: mfspr 12, 528                    # encoding: [0x7d,0x90,0x82,0xa6]
# CHECK-LE: mfspr 12, 528                    # encoding: [0xa6,0x82,0x90,0x7d]
mfibatu %r12, 0
# CHECK-BE: mfspr 12, 529                    # encoding: [0x7d,0x91,0x82,0xa6]
# CHECK-LE: mfspr 12, 529                    # encoding: [0xa6,0x82,0x91,0x7d]
mfibatl %r12, 0
# CHECK-BE: mfspr 12, 530                    # encoding: [0x7d,0x92,0x82,0xa6]
# CHECK-LE: mfspr 12, 530                    # encoding: [0xa6,0x82,0x92,0x7d]
mfibatu %r12, 1
# CHECK-BE: mfspr 12, 531                    # encoding: [0x7d,0x93,0x82,0xa6]
# CHECK-LE: mfspr 12, 531                    # encoding: [0xa6,0x82,0x93,0x7d]
mfibatl %r12, 1
# CHECK-BE: mfspr 12, 532                    # encoding: [0x7d,0x94,0x82,0xa6]
# CHECK-LE: mfspr 12, 532                    # encoding: [0xa6,0x82,0x94,0x7d]
mfibatu %r12, 2
# CHECK-BE: mfspr 12, 533                    # encoding: [0x7d,0x95,0x82,0xa6]
# CHECK-LE: mfspr 12, 533                    # encoding: [0xa6,0x82,0x95,0x7d]
mfibatl %r12, 2
# CHECK-BE: mfspr 12, 534                    # encoding: [0x7d,0x96,0x82,0xa6]
# CHECK-LE: mfspr 12, 534                    # encoding: [0xa6,0x82,0x96,0x7d]
mfibatu %r12, 3
# CHECK-BE: mfspr 12, 535                    # encoding: [0x7d,0x97,0x82,0xa6]
# CHECK-LE: mfspr 12, 535                    # encoding: [0xa6,0x82,0x97,0x7d]
mfibatl %r12, 3
# CHECK-BE: mtspr 528, 12                    # encoding: [0x7d,0x90,0x83,0xa6]
# CHECK-LE: mtspr 528, 12                    # encoding: [0xa6,0x83,0x90,0x7d]
mtibatu 0, %r12
# CHECK-BE: mtspr 529, 12                    # encoding: [0x7d,0x91,0x83,0xa6]
# CHECK-LE: mtspr 529, 12                    # encoding: [0xa6,0x83,0x91,0x7d]
mtibatl 0, %r12
# CHECK-BE: mtspr 530, 12                    # encoding: [0x7d,0x92,0x83,0xa6]
# CHECK-LE: mtspr 530, 12                    # encoding: [0xa6,0x83,0x92,0x7d]
mtibatu 1, %r12
# CHECK-BE: mtspr 531, 12                    # encoding: [0x7d,0x93,0x83,0xa6]
# CHECK-LE: mtspr 531, 12                    # encoding: [0xa6,0x83,0x93,0x7d]
mtibatl 1, %r12
# CHECK-BE: mtspr 532, 12                    # encoding: [0x7d,0x94,0x83,0xa6]
# CHECK-LE: mtspr 532, 12                    # encoding: [0xa6,0x83,0x94,0x7d]
mtibatu 2, %r12
# CHECK-BE: mtspr 533, 12                    # encoding: [0x7d,0x95,0x83,0xa6]
# CHECK-LE: mtspr 533, 12                    # encoding: [0xa6,0x83,0x95,0x7d]
mtibatl 2, %r12
# CHECK-BE: mtspr 534, 12                    # encoding: [0x7d,0x96,0x83,0xa6]
# CHECK-LE: mtspr 534, 12                    # encoding: [0xa6,0x83,0x96,0x7d]
mtibatu 3, %r12
# CHECK-BE: mtspr 535, 12                    # encoding: [0x7d,0x97,0x83,0xa6]
# CHECK-LE: mtspr 535, 12                    # encoding: [0xa6,0x83,0x97,0x7d]
mtibatl 3, %r12

# CHECK-BE: mfspr 12, 536                    # encoding: [0x7d,0x98,0x82,0xa6]
# CHECK-LE: mfspr 12, 536                    # encoding: [0xa6,0x82,0x98,0x7d]
mfdbatu %r12, 0
# CHECK-BE: mfspr 12, 537                    # encoding: [0x7d,0x99,0x82,0xa6]
# CHECK-LE: mfspr 12, 537                    # encoding: [0xa6,0x82,0x99,0x7d]
mfdbatl %r12, 0
# CHECK-BE: mfspr 12, 538                    # encoding: [0x7d,0x9a,0x82,0xa6]
# CHECK-LE: mfspr 12, 538                    # encoding: [0xa6,0x82,0x9a,0x7d]
mfdbatu %r12, 1
# CHECK-BE: mfspr 12, 539                    # encoding: [0x7d,0x9b,0x82,0xa6]
# CHECK-LE: mfspr 12, 539                    # encoding: [0xa6,0x82,0x9b,0x7d]
mfdbatl %r12, 1
# CHECK-BE: mfspr 12, 540                    # encoding: [0x7d,0x9c,0x82,0xa6]
# CHECK-LE: mfspr 12, 540                    # encoding: [0xa6,0x82,0x9c,0x7d]
mfdbatu %r12, 2
# CHECK-BE: mfspr 12, 541                    # encoding: [0x7d,0x9d,0x82,0xa6]
# CHECK-LE: mfspr 12, 541                    # encoding: [0xa6,0x82,0x9d,0x7d]
mfdbatl %r12, 2
# CHECK-BE: mfspr 12, 542                    # encoding: [0x7d,0x9e,0x82,0xa6]
# CHECK-LE: mfspr 12, 542                    # encoding: [0xa6,0x82,0x9e,0x7d]
mfdbatu %r12, 3
# CHECK-BE: mfspr 12, 543                    # encoding: [0x7d,0x9f,0x82,0xa6]
# CHECK-LE: mfspr 12, 543                    # encoding: [0xa6,0x82,0x9f,0x7d]
mfdbatl %r12, 3
# CHECK-BE: mtspr 536, 12                    # encoding: [0x7d,0x98,0x83,0xa6]
# CHECK-LE: mtspr 536, 12                    # encoding: [0xa6,0x83,0x98,0x7d]
mtdbatu 0, %r12
# CHECK-BE: mtspr 537, 12                    # encoding: [0x7d,0x99,0x83,0xa6]
# CHECK-LE: mtspr 537, 12                    # encoding: [0xa6,0x83,0x99,0x7d]
mtdbatl 0, %r12
# CHECK-BE: mtspr 538, 12                    # encoding: [0x7d,0x9a,0x83,0xa6]
# CHECK-LE: mtspr 538, 12                    # encoding: [0xa6,0x83,0x9a,0x7d]
mtdbatu 1, %r12
# CHECK-BE: mtspr 539, 12                    # encoding: [0x7d,0x9b,0x83,0xa6]
# CHECK-LE: mtspr 539, 12                    # encoding: [0xa6,0x83,0x9b,0x7d]
mtdbatl 1, %r12
# CHECK-BE: mtspr 540, 12                    # encoding: [0x7d,0x9c,0x83,0xa6]
# CHECK-LE: mtspr 540, 12                    # encoding: [0xa6,0x83,0x9c,0x7d]
mtdbatu 2, %r12
# CHECK-BE: mtspr 541, 12                    # encoding: [0x7d,0x9d,0x83,0xa6]
# CHECK-LE: mtspr 541, 12                    # encoding: [0xa6,0x83,0x9d,0x7d]
mtdbatl 2, %r12
# CHECK-BE: mtspr 542, 12                    # encoding: [0x7d,0x9e,0x83,0xa6]
# CHECK-LE: mtspr 542, 12                    # encoding: [0xa6,0x83,0x9e,0x7d]
mtdbatu 3, %r12
# CHECK-BE: mtspr 543, 12                    # encoding: [0x7d,0x9f,0x83,0xa6]
# CHECK-LE: mtspr 543, 12                    # encoding: [0xa6,0x83,0x9f,0x7d]
mtdbatl 3, %r12

# CHECK-BE: tlbld 4                        # encoding: [0x7c,0x00,0x27,0xa4]
# CHECK-LE: tlbld 4                        # encoding: [0xa4,0x27,0x00,0x7c]
tlbld %r4
# CHECK-BE: tlbli 4                        # encoding: [0x7c,0x00,0x27,0xe4]
# CHECK-LE: tlbli 4                        # encoding: [0xe4,0x27,0x00,0x7c]
tlbli %r4
