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

# CHECK-BE: mfspr 2, 1018                    # encoding: [0x7c,0x5a,0xfa,0xa6]
# CHECK-LE: mfspr 2, 1018                    # encoding: [0xa6,0xfa,0x5a,0x7c]
            mfdccr %r2
# CHECK-BE: mtspr 1018, 2                    # encoding: [0x7c,0x5a,0xfb,0xa6]
# CHECK-LE: mtspr 1018, 2                    # encoding: [0xa6,0xfb,0x5a,0x7c]
            mtdccr %r2

# CHECK-BE: mfspr 2, 1019                    # encoding: [0x7c,0x5b,0xfa,0xa6]
# CHECK-LE: mfspr 2, 1019                    # encoding: [0xa6,0xfa,0x5b,0x7c]
            mficcr %r2
# CHECK-BE: mtspr 1019, 2                    # encoding: [0x7c,0x5b,0xfb,0xa6]
# CHECK-LE: mtspr 1019, 2                    # encoding: [0xa6,0xfb,0x5b,0x7c]
            mticcr %r2

# CHECK-BE: mfspr 2, 981                    # encoding: [0x7c,0x55,0xf2,0xa6]
# CHECK-LE: mfspr 2, 981                    # encoding: [0xa6,0xf2,0x55,0x7c]
            mfdear %r2
# CHECK-BE: mtspr 981, 2                    # encoding: [0x7c,0x55,0xf3,0xa6]
# CHECK-LE: mtspr 981, 2                    # encoding: [0xa6,0xf3,0x55,0x7c]
            mtdear %r2

# CHECK-BE: mfspr 2, 980                    # encoding: [0x7c,0x54,0xf2,0xa6]
# CHECK-LE: mfspr 2, 980                    # encoding: [0xa6,0xf2,0x54,0x7c]
            mfesr %r2
# CHECK-BE: mtspr 980, 2                    # encoding: [0x7c,0x54,0xf3,0xa6]
# CHECK-LE: mtspr 980, 2                    # encoding: [0xa6,0xf3,0x54,0x7c]
            mtesr %r2

# CHECK-BE: mfspr 2, 986                    # encoding: [0x7c,0x5a,0xf2,0xa6]
# CHECK-LE: mfspr 2, 986                    # encoding: [0xa6,0xf2,0x5a,0x7c]
            mftcr %r2
# CHECK-BE: mtspr 986, 2                    # encoding: [0x7c,0x5a,0xf3,0xa6]
# CHECK-LE: mtspr 986, 2                    # encoding: [0xa6,0xf3,0x5a,0x7c]
            mttcr %r2

# CHECK-BE: mfspr 2, 989                    # encoding: [0x7c,0x5d,0xf2,0xa6]
# CHECK-LE: mfspr 2, 989                    # encoding: [0xa6,0xf2,0x5d,0x7c]
            mftblo %r2
# CHECK-BE: mtspr 989, 2                    # encoding: [0x7c,0x5d,0xf3,0xa6]
# CHECK-LE: mtspr 989, 2                    # encoding: [0xa6,0xf3,0x5d,0x7c]
            mttblo %r2
# CHECK-BE: mfspr 2, 988                    # encoding: [0x7c,0x5c,0xf2,0xa6]
# CHECK-LE: mfspr 2, 988                    # encoding: [0xa6,0xf2,0x5c,0x7c]
            mftbhi %r2
# CHECK-BE: mtspr 988, 2                    # encoding: [0x7c,0x5c,0xf3,0xa6]
# CHECK-LE: mtspr 988, 2                    # encoding: [0xa6,0xf3,0x5c,0x7c]
            mttbhi %r2

# CHECK-BE: dccci 5, 6                      # encoding: [0x7c,0x05,0x33,0x8c]
# CHECK-LE: dccci 5, 6                      # encoding: [0x8c,0x33,0x05,0x7c]
            dccci %r5,%r6
# CHECK-BE: iccci 5, 6                      # encoding: [0x7c,0x05,0x37,0x8c]
# CHECK-LE: iccci 5, 6                      # encoding: [0x8c,0x37,0x05,0x7c]
            iccci %r5,%r6
# CHECK-BE: dccci 0, 0                      # encoding: [0x7c,0x00,0x03,0x8c]
# CHECK-LE: dccci 0, 0                      # encoding: [0x8c,0x03,0x00,0x7c]
            dci %r0
# CHECK-BE: iccci 0, 0                      # encoding: [0x7c,0x00,0x07,0x8c]
# CHECK-LE: iccci 0, 0                      # encoding: [0x8c,0x07,0x00,0x7c]
            ici 0

# CHECK-BE: mfspr 2, 990                    # encoding: [0x7c,0x5e,0xf2,0xa6]
# CHECK-LE: mfspr 2, 990                    # encoding: [0xa6,0xf2,0x5e,0x7c]
            mfsrr2 2
# CHECK-BE: mtspr 990, 2                    # encoding: [0x7c,0x5e,0xf3,0xa6]
# CHECK-LE: mtspr 990, 2                    # encoding: [0xa6,0xf3,0x5e,0x7c]
            mtsrr2 2
# CHECK-BE: mfspr 2, 991                    # encoding: [0x7c,0x5f,0xf2,0xa6]
# CHECK-LE: mfspr 2, 991                    # encoding: [0xa6,0xf2,0x5f,0x7c]
            mfsrr3 2
# CHECK-BE: mtspr 991, 2                    # encoding: [0x7c,0x5f,0xf3,0xa6]
# CHECK-LE: mtspr 991, 2                    # encoding: [0xa6,0xf3,0x5f,0x7c]
            mtsrr3 2

# CHECK-BE: mfdcr 5, 128                    # encoding: [0x7c,0xa0,0x22,0x86]
# CHECK-LE: mfdcr 5, 128                    # encoding: [0x86,0x22,0xa0,0x7c]
            mfbr0 %r5
# CHECK-BE: mtdcr 128, 5                    # encoding: [0x7c,0xa0,0x23,0x86]
# CHECK-LE: mtdcr 128, 5                    # encoding: [0x86,0x23,0xa0,0x7c]
            mtbr0 %r5
# CHECK-BE: mfdcr 5, 129                    # encoding: [0x7c,0xa1,0x22,0x86]
# CHECK-LE: mfdcr 5, 129                    # encoding: [0x86,0x22,0xa1,0x7c]
            mfbr1 %r5
# CHECK-BE: mtdcr 129, 5                    # encoding: [0x7c,0xa1,0x23,0x86]
# CHECK-LE: mtdcr 129, 5                    # encoding: [0x86,0x23,0xa1,0x7c]
            mtbr1 %r5
# CHECK-BE: mfdcr 5, 130                    # encoding: [0x7c,0xa2,0x22,0x86]
# CHECK-LE: mfdcr 5, 130                    # encoding: [0x86,0x22,0xa2,0x7c]
            mfbr2 %r5
# CHECK-BE: mtdcr 130, 5                    # encoding: [0x7c,0xa2,0x23,0x86]
# CHECK-LE: mtdcr 130, 5                    # encoding: [0x86,0x23,0xa2,0x7c]
            mtbr2 %r5
# CHECK-BE: mfdcr 5, 131                    # encoding: [0x7c,0xa3,0x22,0x86]
# CHECK-LE: mfdcr 5, 131                    # encoding: [0x86,0x22,0xa3,0x7c]
            mfbr3 %r5
# CHECK-BE: mtdcr 131, 5                    # encoding: [0x7c,0xa3,0x23,0x86]
# CHECK-LE: mtdcr 131, 5                    # encoding: [0x86,0x23,0xa3,0x7c]
            mtbr3 %r5
# CHECK-BE: mfdcr 5, 132                    # encoding: [0x7c,0xa4,0x22,0x86]
# CHECK-LE: mfdcr 5, 132                    # encoding: [0x86,0x22,0xa4,0x7c]
            mfbr4 %r5
# CHECK-BE: mtdcr 132, 5                    # encoding: [0x7c,0xa4,0x23,0x86]
# CHECK-LE: mtdcr 132, 5                    # encoding: [0x86,0x23,0xa4,0x7c]
            mtbr4 %r5
# CHECK-BE: mfdcr 5, 133                    # encoding: [0x7c,0xa5,0x22,0x86]
# CHECK-LE: mfdcr 5, 133                    # encoding: [0x86,0x22,0xa5,0x7c]
            mfbr5 %r5
# CHECK-BE: mtdcr 133, 5                    # encoding: [0x7c,0xa5,0x23,0x86]
# CHECK-LE: mtdcr 133, 5                    # encoding: [0x86,0x23,0xa5,0x7c]
            mtbr5 %r5
# CHECK-BE: mfdcr 5, 134                    # encoding: [0x7c,0xa6,0x22,0x86]
# CHECK-LE: mfdcr 5, 134                    # encoding: [0x86,0x22,0xa6,0x7c]
            mfbr6 %r5
# CHECK-BE: mtdcr 134, 5                    # encoding: [0x7c,0xa6,0x23,0x86]
# CHECK-LE: mtdcr 134, 5                    # encoding: [0x86,0x23,0xa6,0x7c]
            mtbr6 %r5
# CHECK-BE: mfdcr 5, 135                    # encoding: [0x7c,0xa7,0x22,0x86]
# CHECK-LE: mfdcr 5, 135                    # encoding: [0x86,0x22,0xa7,0x7c]
            mfbr7 %r5
# CHECK-BE: mtdcr 135, 5                    # encoding: [0x7c,0xa7,0x23,0x86]
# CHECK-LE: mtdcr 135, 5                    # encoding: [0x86,0x23,0xa7,0x7c]
            mtbr7 %r5
