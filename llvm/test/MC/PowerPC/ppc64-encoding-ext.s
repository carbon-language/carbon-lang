
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Condition register bit symbols

# CHECK-BE: beqlr 0                         # encoding: [0x4d,0x82,0x00,0x20]
# CHECK-LE: beqlr 0                         # encoding: [0x20,0x00,0x82,0x4d]
            beqlr cr0
# CHECK-BE: beqlr 1                         # encoding: [0x4d,0x86,0x00,0x20]
# CHECK-LE: beqlr 1                         # encoding: [0x20,0x00,0x86,0x4d]
            beqlr cr1
# CHECK-BE: beqlr 2                         # encoding: [0x4d,0x8a,0x00,0x20]
# CHECK-LE: beqlr 2                         # encoding: [0x20,0x00,0x8a,0x4d]
            beqlr cr2
# CHECK-BE: beqlr 3                         # encoding: [0x4d,0x8e,0x00,0x20]
# CHECK-LE: beqlr 3                         # encoding: [0x20,0x00,0x8e,0x4d]
            beqlr cr3
# CHECK-BE: beqlr 4                         # encoding: [0x4d,0x92,0x00,0x20]
# CHECK-LE: beqlr 4                         # encoding: [0x20,0x00,0x92,0x4d]
            beqlr cr4
# CHECK-BE: beqlr 5                         # encoding: [0x4d,0x96,0x00,0x20]
# CHECK-LE: beqlr 5                         # encoding: [0x20,0x00,0x96,0x4d]
            beqlr cr5
# CHECK-BE: beqlr 6                         # encoding: [0x4d,0x9a,0x00,0x20]
# CHECK-LE: beqlr 6                         # encoding: [0x20,0x00,0x9a,0x4d]
            beqlr cr6
# CHECK-BE: beqlr 7                         # encoding: [0x4d,0x9e,0x00,0x20]
# CHECK-LE: beqlr 7                         # encoding: [0x20,0x00,0x9e,0x4d]
            beqlr cr7

# CHECK-BE: bclr 12, 0, 0                   # encoding: [0x4d,0x80,0x00,0x20]
# CHECK-LE: bclr 12, 0, 0                   # encoding: [0x20,0x00,0x80,0x4d]
            btlr 4*cr0+lt
# CHECK-BE: bclr 12, 1, 0                   # encoding: [0x4d,0x81,0x00,0x20]
# CHECK-LE: bclr 12, 1, 0                   # encoding: [0x20,0x00,0x81,0x4d]
            btlr 4*cr0+gt
# CHECK-BE: bclr 12, 2, 0                   # encoding: [0x4d,0x82,0x00,0x20]
# CHECK-LE: bclr 12, 2, 0                   # encoding: [0x20,0x00,0x82,0x4d]
            btlr 4*cr0+eq
# CHECK-BE: bclr 12, 3, 0                   # encoding: [0x4d,0x83,0x00,0x20]
# CHECK-LE: bclr 12, 3, 0                   # encoding: [0x20,0x00,0x83,0x4d]
            btlr 4*cr0+so
# CHECK-BE: bclr 12, 3, 0                   # encoding: [0x4d,0x83,0x00,0x20]
# CHECK-LE: bclr 12, 3, 0                   # encoding: [0x20,0x00,0x83,0x4d]
            btlr 4*cr0+un
# CHECK-BE: bclr 12, 4, 0                   # encoding: [0x4d,0x84,0x00,0x20]
# CHECK-LE: bclr 12, 4, 0                   # encoding: [0x20,0x00,0x84,0x4d]
            btlr 4*cr1+lt
# CHECK-BE: bclr 12, 5, 0                   # encoding: [0x4d,0x85,0x00,0x20]
# CHECK-LE: bclr 12, 5, 0                   # encoding: [0x20,0x00,0x85,0x4d]
            btlr 4*cr1+gt
# CHECK-BE: bclr 12, 6, 0                   # encoding: [0x4d,0x86,0x00,0x20]
# CHECK-LE: bclr 12, 6, 0                   # encoding: [0x20,0x00,0x86,0x4d]
            btlr 4*cr1+eq
# CHECK-BE: bclr 12, 7, 0                   # encoding: [0x4d,0x87,0x00,0x20]
# CHECK-LE: bclr 12, 7, 0                   # encoding: [0x20,0x00,0x87,0x4d]
            btlr 4*cr1+so
# CHECK-BE: bclr 12, 7, 0                   # encoding: [0x4d,0x87,0x00,0x20]
# CHECK-LE: bclr 12, 7, 0                   # encoding: [0x20,0x00,0x87,0x4d]
            btlr 4*cr1+un
# CHECK-BE: bclr 12, 8, 0                   # encoding: [0x4d,0x88,0x00,0x20]
# CHECK-LE: bclr 12, 8, 0                   # encoding: [0x20,0x00,0x88,0x4d]
            btlr 4*cr2+lt
# CHECK-BE: bclr 12, 9, 0                   # encoding: [0x4d,0x89,0x00,0x20]
# CHECK-LE: bclr 12, 9, 0                   # encoding: [0x20,0x00,0x89,0x4d]
            btlr 4*cr2+gt
# CHECK-BE: bclr 12, 10, 0                  # encoding: [0x4d,0x8a,0x00,0x20]
# CHECK-LE: bclr 12, 10, 0                  # encoding: [0x20,0x00,0x8a,0x4d]
            btlr 4*cr2+eq
# CHECK-BE: bclr 12, 11, 0                  # encoding: [0x4d,0x8b,0x00,0x20]
# CHECK-LE: bclr 12, 11, 0                  # encoding: [0x20,0x00,0x8b,0x4d]
            btlr 4*cr2+so
# CHECK-BE: bclr 12, 11, 0                  # encoding: [0x4d,0x8b,0x00,0x20]
# CHECK-LE: bclr 12, 11, 0                  # encoding: [0x20,0x00,0x8b,0x4d]
            btlr 4*cr2+un
# CHECK-BE: bclr 12, 12, 0                  # encoding: [0x4d,0x8c,0x00,0x20]
# CHECK-LE: bclr 12, 12, 0                  # encoding: [0x20,0x00,0x8c,0x4d]
            btlr 4*cr3+lt
# CHECK-BE: bclr 12, 13, 0                  # encoding: [0x4d,0x8d,0x00,0x20]
# CHECK-LE: bclr 12, 13, 0                  # encoding: [0x20,0x00,0x8d,0x4d]
            btlr 4*cr3+gt
# CHECK-BE: bclr 12, 14, 0                  # encoding: [0x4d,0x8e,0x00,0x20]
# CHECK-LE: bclr 12, 14, 0                  # encoding: [0x20,0x00,0x8e,0x4d]
            btlr 4*cr3+eq
# CHECK-BE: bclr 12, 15, 0                  # encoding: [0x4d,0x8f,0x00,0x20]
# CHECK-LE: bclr 12, 15, 0                  # encoding: [0x20,0x00,0x8f,0x4d]
            btlr 4*cr3+so
# CHECK-BE: bclr 12, 15, 0                  # encoding: [0x4d,0x8f,0x00,0x20]
# CHECK-LE: bclr 12, 15, 0                  # encoding: [0x20,0x00,0x8f,0x4d]
            btlr 4*cr3+un
# CHECK-BE: bclr 12, 16, 0                  # encoding: [0x4d,0x90,0x00,0x20]
# CHECK-LE: bclr 12, 16, 0                  # encoding: [0x20,0x00,0x90,0x4d]
            btlr 4*cr4+lt
# CHECK-BE: bclr 12, 17, 0                  # encoding: [0x4d,0x91,0x00,0x20]
# CHECK-LE: bclr 12, 17, 0                  # encoding: [0x20,0x00,0x91,0x4d]
            btlr 4*cr4+gt
# CHECK-BE: bclr 12, 18, 0                  # encoding: [0x4d,0x92,0x00,0x20]
# CHECK-LE: bclr 12, 18, 0                  # encoding: [0x20,0x00,0x92,0x4d]
            btlr 4*cr4+eq
# CHECK-BE: bclr 12, 19, 0                  # encoding: [0x4d,0x93,0x00,0x20]
# CHECK-LE: bclr 12, 19, 0                  # encoding: [0x20,0x00,0x93,0x4d]
            btlr 4*cr4+so
# CHECK-BE: bclr 12, 19, 0                  # encoding: [0x4d,0x93,0x00,0x20]
# CHECK-LE: bclr 12, 19, 0                  # encoding: [0x20,0x00,0x93,0x4d]
            btlr 4*cr4+un
# CHECK-BE: bclr 12, 20, 0                  # encoding: [0x4d,0x94,0x00,0x20]
# CHECK-LE: bclr 12, 20, 0                  # encoding: [0x20,0x00,0x94,0x4d]
            btlr 4*cr5+lt
# CHECK-BE: bclr 12, 21, 0                  # encoding: [0x4d,0x95,0x00,0x20]
# CHECK-LE: bclr 12, 21, 0                  # encoding: [0x20,0x00,0x95,0x4d]
            btlr 4*cr5+gt
# CHECK-BE: bclr 12, 22, 0                  # encoding: [0x4d,0x96,0x00,0x20]
# CHECK-LE: bclr 12, 22, 0                  # encoding: [0x20,0x00,0x96,0x4d]
            btlr 4*cr5+eq
# CHECK-BE: bclr 12, 23, 0                  # encoding: [0x4d,0x97,0x00,0x20]
# CHECK-LE: bclr 12, 23, 0                  # encoding: [0x20,0x00,0x97,0x4d]
            btlr 4*cr5+so
# CHECK-BE: bclr 12, 23, 0                  # encoding: [0x4d,0x97,0x00,0x20]
# CHECK-LE: bclr 12, 23, 0                  # encoding: [0x20,0x00,0x97,0x4d]
            btlr 4*cr5+un
# CHECK-BE: bclr 12, 24, 0                  # encoding: [0x4d,0x98,0x00,0x20]
# CHECK-LE: bclr 12, 24, 0                  # encoding: [0x20,0x00,0x98,0x4d]
            btlr 4*cr6+lt
# CHECK-BE: bclr 12, 25, 0                  # encoding: [0x4d,0x99,0x00,0x20]
# CHECK-LE: bclr 12, 25, 0                  # encoding: [0x20,0x00,0x99,0x4d]
            btlr 4*cr6+gt
# CHECK-BE: bclr 12, 26, 0                  # encoding: [0x4d,0x9a,0x00,0x20]
# CHECK-LE: bclr 12, 26, 0                  # encoding: [0x20,0x00,0x9a,0x4d]
            btlr 4*cr6+eq
# CHECK-BE: bclr 12, 27, 0                  # encoding: [0x4d,0x9b,0x00,0x20]
# CHECK-LE: bclr 12, 27, 0                  # encoding: [0x20,0x00,0x9b,0x4d]
            btlr 4*cr6+so
# CHECK-BE: bclr 12, 27, 0                  # encoding: [0x4d,0x9b,0x00,0x20]
# CHECK-LE: bclr 12, 27, 0                  # encoding: [0x20,0x00,0x9b,0x4d]
            btlr 4*cr6+un
# CHECK-BE: bclr 12, 28, 0                  # encoding: [0x4d,0x9c,0x00,0x20]
# CHECK-LE: bclr 12, 28, 0                  # encoding: [0x20,0x00,0x9c,0x4d]
            btlr 4*cr7+lt
# CHECK-BE: bclr 12, 29, 0                  # encoding: [0x4d,0x9d,0x00,0x20]
# CHECK-LE: bclr 12, 29, 0                  # encoding: [0x20,0x00,0x9d,0x4d]
            btlr 4*cr7+gt
# CHECK-BE: bclr 12, 30, 0                  # encoding: [0x4d,0x9e,0x00,0x20]
# CHECK-LE: bclr 12, 30, 0                  # encoding: [0x20,0x00,0x9e,0x4d]
            btlr 4*cr7+eq
# CHECK-BE: bclr 12, 31, 0                  # encoding: [0x4d,0x9f,0x00,0x20]
# CHECK-LE: bclr 12, 31, 0                  # encoding: [0x20,0x00,0x9f,0x4d]
            btlr 4*cr7+so
# CHECK-BE: bclr 12, 31, 0                  # encoding: [0x4d,0x9f,0x00,0x20]
# CHECK-LE: bclr 12, 31, 0                  # encoding: [0x20,0x00,0x9f,0x4d]
            btlr 4*cr7+un

# Branch mnemonics

# CHECK-BE: blr                             # encoding: [0x4e,0x80,0x00,0x20]
# CHECK-LE: blr                             # encoding: [0x20,0x00,0x80,0x4e]
            blr
# CHECK-BE: bctr                            # encoding: [0x4e,0x80,0x04,0x20]
# CHECK-LE: bctr                            # encoding: [0x20,0x04,0x80,0x4e]
            bctr
# CHECK-BE: blrl                            # encoding: [0x4e,0x80,0x00,0x21]
# CHECK-LE: blrl                            # encoding: [0x21,0x00,0x80,0x4e]
            blrl
# CHECK-BE: bctrl                           # encoding: [0x4e,0x80,0x04,0x21]
# CHECK-LE: bctrl                           # encoding: [0x21,0x04,0x80,0x4e]
            bctrl

# CHECK-BE: bc 12, 2, target                # encoding: [0x41,0x82,A,0bAAAAAA00]
# CHECK-LE: bc 12, 2, target                # encoding: [0bAAAAAA00,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bt 2, target
# CHECK-BE: bca 12, 2, target               # encoding: [0x41,0x82,A,0bAAAAAA10]
# CHECK-LE: bca 12, 2, target               # encoding: [0bAAAAAA10,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bta 2, target
# CHECK-BE: bclr 12, 2, 0                   # encoding: [0x4d,0x82,0x00,0x20]
# CHECK-LE: bclr 12, 2, 0                   # encoding: [0x20,0x00,0x82,0x4d]
            btlr 2
# CHECK-BE: bcctr 12, 2, 0                  # encoding: [0x4d,0x82,0x04,0x20]
# CHECK-LE: bcctr 12, 2, 0                  # encoding: [0x20,0x04,0x82,0x4d]
            btctr 2
# CHECK-BE: bcl 12, 2, target               # encoding: [0x41,0x82,A,0bAAAAAA01]
# CHECK-LE: bcl 12, 2, target               # encoding: [0bAAAAAA01,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            btl 2, target
# CHECK-BE: bcla 12, 2, target              # encoding: [0x41,0x82,A,0bAAAAAA11]
# CHECK-LE: bcla 12, 2, target              # encoding: [0bAAAAAA11,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            btla 2, target
# CHECK-BE: bclrl 12, 2, 0                  # encoding: [0x4d,0x82,0x00,0x21]
# CHECK-LE: bclrl 12, 2, 0                  # encoding: [0x21,0x00,0x82,0x4d]
            btlrl 2
# CHECK-BE: bcctrl 12, 2, 0                 # encoding: [0x4d,0x82,0x04,0x21]
# CHECK-LE: bcctrl 12, 2, 0                 # encoding: [0x21,0x04,0x82,0x4d]
            btctrl 2

# CHECK-BE: bc 15, 2, target                # encoding: [0x41,0xe2,A,0bAAAAAA00]
# CHECK-LE: bc 15, 2, target                # encoding: [0bAAAAAA00,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bt+ 2, target
# CHECK-BE: bca 15, 2, target               # encoding: [0x41,0xe2,A,0bAAAAAA10]
# CHECK-LE: bca 15, 2, target               # encoding: [0bAAAAAA10,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bta+ 2, target
# CHECK-BE: bclr 15, 2, 0                   # encoding: [0x4d,0xe2,0x00,0x20]
# CHECK-LE: bclr 15, 2, 0                   # encoding: [0x20,0x00,0xe2,0x4d]
            btlr+ 2
# CHECK-BE: bcctr 15, 2, 0                  # encoding: [0x4d,0xe2,0x04,0x20]
# CHECK-LE: bcctr 15, 2, 0                  # encoding: [0x20,0x04,0xe2,0x4d]
            btctr+ 2
# CHECK-BE: bcl 15, 2, target               # encoding: [0x41,0xe2,A,0bAAAAAA01]
# CHECK-LE: bcl 15, 2, target               # encoding: [0bAAAAAA01,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            btl+ 2, target
# CHECK-BE: bcla 15, 2, target              # encoding: [0x41,0xe2,A,0bAAAAAA11]
# CHECK-LE: bcla 15, 2, target              # encoding: [0bAAAAAA11,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            btla+ 2, target
# CHECK-BE: bclrl 15, 2, 0                  # encoding: [0x4d,0xe2,0x00,0x21]
# CHECK-LE: bclrl 15, 2, 0                  # encoding: [0x21,0x00,0xe2,0x4d]
            btlrl+ 2
# CHECK-BE: bcctrl 15, 2, 0                 # encoding: [0x4d,0xe2,0x04,0x21]
# CHECK-LE: bcctrl 15, 2, 0                 # encoding: [0x21,0x04,0xe2,0x4d]
            btctrl+ 2

# CHECK-BE: bc 14, 2, target                # encoding: [0x41,0xc2,A,0bAAAAAA00]
# CHECK-LE: bc 14, 2, target                # encoding: [0bAAAAAA00,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bt- 2, target
# CHECK-BE: bca 14, 2, target               # encoding: [0x41,0xc2,A,0bAAAAAA10]
# CHECK-LE: bca 14, 2, target               # encoding: [0bAAAAAA10,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bta- 2, target
# CHECK-BE: bclr 14, 2, 0                   # encoding: [0x4d,0xc2,0x00,0x20]
# CHECK-LE: bclr 14, 2, 0                   # encoding: [0x20,0x00,0xc2,0x4d]
            btlr- 2
# CHECK-BE: bcctr 14, 2, 0                  # encoding: [0x4d,0xc2,0x04,0x20]
# CHECK-LE: bcctr 14, 2, 0                  # encoding: [0x20,0x04,0xc2,0x4d]
            btctr- 2
# CHECK-BE: bcl 14, 2, target               # encoding: [0x41,0xc2,A,0bAAAAAA01]
# CHECK-LE: bcl 14, 2, target               # encoding: [0bAAAAAA01,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            btl- 2, target
# CHECK-BE: bcla 14, 2, target              # encoding: [0x41,0xc2,A,0bAAAAAA11]
# CHECK-LE: bcla 14, 2, target              # encoding: [0bAAAAAA11,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            btla- 2, target
# CHECK-BE: bclrl 14, 2, 0                  # encoding: [0x4d,0xc2,0x00,0x21]
# CHECK-LE: bclrl 14, 2, 0                  # encoding: [0x21,0x00,0xc2,0x4d]
            btlrl- 2
# CHECK-BE: bcctrl 14, 2, 0                 # encoding: [0x4d,0xc2,0x04,0x21]
# CHECK-LE: bcctrl 14, 2, 0                 # encoding: [0x21,0x04,0xc2,0x4d]
            btctrl- 2

# CHECK-BE: bc 4, 2, target                 # encoding: [0x40,0x82,A,0bAAAAAA00]
# CHECK-LE: bc 4, 2, target                 # encoding: [0bAAAAAA00,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bf 2, target
# CHECK-BE: bca 4, 2, target                # encoding: [0x40,0x82,A,0bAAAAAA10]
# CHECK-LE: bca 4, 2, target                # encoding: [0bAAAAAA10,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfa 2, target
# CHECK-BE: bclr 4, 2, 0                    # encoding: [0x4c,0x82,0x00,0x20]
# CHECK-LE: bclr 4, 2, 0                    # encoding: [0x20,0x00,0x82,0x4c]
            bflr 2
# CHECK-BE: bcctr 4, 2, 0                   # encoding: [0x4c,0x82,0x04,0x20]
# CHECK-LE: bcctr 4, 2, 0                   # encoding: [0x20,0x04,0x82,0x4c]
            bfctr 2
# CHECK-BE: bcl 4, 2, target                # encoding: [0x40,0x82,A,0bAAAAAA01]
# CHECK-LE: bcl 4, 2, target                # encoding: [0bAAAAAA01,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bfl 2, target
# CHECK-BE: bcla 4, 2, target               # encoding: [0x40,0x82,A,0bAAAAAA11]
# CHECK-LE: bcla 4, 2, target               # encoding: [0bAAAAAA11,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfla 2, target
# CHECK-BE: bclrl 4, 2, 0                   # encoding: [0x4c,0x82,0x00,0x21]
# CHECK-LE: bclrl 4, 2, 0                   # encoding: [0x21,0x00,0x82,0x4c]
            bflrl 2
# CHECK-BE: bcctrl 4, 2, 0                  # encoding: [0x4c,0x82,0x04,0x21]
# CHECK-LE: bcctrl 4, 2, 0                  # encoding: [0x21,0x04,0x82,0x4c]
            bfctrl 2

# CHECK-BE: bc 7, 2, target                 # encoding: [0x40,0xe2,A,0bAAAAAA00]
# CHECK-LE: bc 7, 2, target                 # encoding: [0bAAAAAA00,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bf+ 2, target
# CHECK-BE: bca 7, 2, target                # encoding: [0x40,0xe2,A,0bAAAAAA10]
# CHECK-LE: bca 7, 2, target                # encoding: [0bAAAAAA10,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfa+ 2, target
# CHECK-BE: bclr 7, 2, 0                    # encoding: [0x4c,0xe2,0x00,0x20]
# CHECK-LE: bclr 7, 2, 0                    # encoding: [0x20,0x00,0xe2,0x4c]
            bflr+ 2
# CHECK-BE: bcctr 7, 2, 0                   # encoding: [0x4c,0xe2,0x04,0x20]
# CHECK-LE: bcctr 7, 2, 0                   # encoding: [0x20,0x04,0xe2,0x4c]
            bfctr+ 2
# CHECK-BE: bcl 7, 2, target                # encoding: [0x40,0xe2,A,0bAAAAAA01]
# CHECK-LE: bcl 7, 2, target                # encoding: [0bAAAAAA01,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bfl+ 2, target
# CHECK-BE: bcla 7, 2, target               # encoding: [0x40,0xe2,A,0bAAAAAA11]
# CHECK-LE: bcla 7, 2, target               # encoding: [0bAAAAAA11,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfla+ 2, target
# CHECK-BE: bclrl 7, 2, 0                   # encoding: [0x4c,0xe2,0x00,0x21]
# CHECK-LE: bclrl 7, 2, 0                   # encoding: [0x21,0x00,0xe2,0x4c]
            bflrl+ 2
# CHECK-BE: bcctrl 7, 2, 0                  # encoding: [0x4c,0xe2,0x04,0x21]
# CHECK-LE: bcctrl 7, 2, 0                  # encoding: [0x21,0x04,0xe2,0x4c]
            bfctrl+ 2

# CHECK-BE: bc 6, 2, target                 # encoding: [0x40,0xc2,A,0bAAAAAA00]
# CHECK-LE: bc 6, 2, target                 # encoding: [0bAAAAAA00,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bf- 2, target
# CHECK-BE: bca 6, 2, target                # encoding: [0x40,0xc2,A,0bAAAAAA10]
# CHECK-LE: bca 6, 2, target                # encoding: [0bAAAAAA10,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfa- 2, target
# CHECK-BE: bclr 6, 2, 0                    # encoding: [0x4c,0xc2,0x00,0x20]
# CHECK-LE: bclr 6, 2, 0                    # encoding: [0x20,0x00,0xc2,0x4c]
            bflr- 2
# CHECK-BE: bcctr 6, 2, 0                   # encoding: [0x4c,0xc2,0x04,0x20]
# CHECK-LE: bcctr 6, 2, 0                   # encoding: [0x20,0x04,0xc2,0x4c]
            bfctr- 2
# CHECK-BE: bcl 6, 2, target                # encoding: [0x40,0xc2,A,0bAAAAAA01]
# CHECK-LE: bcl 6, 2, target                # encoding: [0bAAAAAA01,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bfl- 2, target
# CHECK-BE: bcla 6, 2, target               # encoding: [0x40,0xc2,A,0bAAAAAA11]
# CHECK-LE: bcla 6, 2, target               # encoding: [0bAAAAAA11,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfla- 2, target
# CHECK-BE: bclrl 6, 2, 0                   # encoding: [0x4c,0xc2,0x00,0x21]
# CHECK-LE: bclrl 6, 2, 0                   # encoding: [0x21,0x00,0xc2,0x4c]
            bflrl- 2
# CHECK-BE: bcctrl 6, 2, 0                  # encoding: [0x4c,0xc2,0x04,0x21]
# CHECK-LE: bcctrl 6, 2, 0                  # encoding: [0x21,0x04,0xc2,0x4c]
            bfctrl- 2

# CHECK-BE: bdnz target                     # encoding: [0x42,0x00,A,0bAAAAAA00]
# CHECK-LE: bdnz target                     # encoding: [0bAAAAAA00,A,0x00,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnz target
# CHECK-BE: bdnza target                    # encoding: [0x42,0x00,A,0bAAAAAA10]
# CHECK-LE: bdnza target                    # encoding: [0bAAAAAA10,A,0x00,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnza target
# CHECK-BE: bdnzlr                          # encoding: [0x4e,0x00,0x00,0x20]
# CHECK-LE: bdnzlr                          # encoding: [0x20,0x00,0x00,0x4e]
            bdnzlr
# CHECK-BE: bdnzl target                    # encoding: [0x42,0x00,A,0bAAAAAA01]
# CHECK-LE: bdnzl target                    # encoding: [0bAAAAAA01,A,0x00,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzl target
# CHECK-BE: bdnzla target                   # encoding: [0x42,0x00,A,0bAAAAAA11]
# CHECK-LE: bdnzla target                   # encoding: [0bAAAAAA11,A,0x00,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzla target
# CHECK-BE: bdnzlrl                         # encoding: [0x4e,0x00,0x00,0x21]
# CHECK-LE: bdnzlrl                         # encoding: [0x21,0x00,0x00,0x4e]
            bdnzlrl

# CHECK-BE: bdnz+ target                    # encoding: [0x43,0x20,A,0bAAAAAA00]
# CHECK-LE: bdnz+ target                    # encoding: [0bAAAAAA00,A,0x20,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnz+ target
# CHECK-BE: bdnza+ target                   # encoding: [0x43,0x20,A,0bAAAAAA10]
# CHECK-LE: bdnza+ target                   # encoding: [0bAAAAAA10,A,0x20,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnza+ target
# CHECK-BE: bdnzlr+                         # encoding: [0x4f,0x20,0x00,0x20]
# CHECK-LE: bdnzlr+                         # encoding: [0x20,0x00,0x20,0x4f]
            bdnzlr+
# CHECK-BE: bdnzl+ target                   # encoding: [0x43,0x20,A,0bAAAAAA01]
# CHECK-LE: bdnzl+ target                   # encoding: [0bAAAAAA01,A,0x20,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzl+ target
# CHECK-BE: bdnzla+ target                  # encoding: [0x43,0x20,A,0bAAAAAA11]
# CHECK-LE: bdnzla+ target                  # encoding: [0bAAAAAA11,A,0x20,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzla+ target
# CHECK-BE: bdnzlrl+                        # encoding: [0x4f,0x20,0x00,0x21]
# CHECK-LE: bdnzlrl+                        # encoding: [0x21,0x00,0x20,0x4f]
            bdnzlrl+

# CHECK-BE: bdnz- target                    # encoding: [0x43,0x00,A,0bAAAAAA00]
# CHECK-LE: bdnz- target                    # encoding: [0bAAAAAA00,A,0x00,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnz- target
# CHECK-BE: bdnza- target                   # encoding: [0x43,0x00,A,0bAAAAAA10]
# CHECK-LE: bdnza- target                   # encoding: [0bAAAAAA10,A,0x00,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnza- target
# CHECK-BE: bdnzlr-                         # encoding: [0x4f,0x00,0x00,0x20]
# CHECK-LE: bdnzlr-                         # encoding: [0x20,0x00,0x00,0x4f]
            bdnzlr-
# CHECK-BE: bdnzl- target                   # encoding: [0x43,0x00,A,0bAAAAAA01]
# CHECK-LE: bdnzl- target                   # encoding: [0bAAAAAA01,A,0x00,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzl- target
# CHECK-BE: bdnzla- target                  # encoding: [0x43,0x00,A,0bAAAAAA11]
# CHECK-LE: bdnzla- target                  # encoding: [0bAAAAAA11,A,0x00,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzla- target
# CHECK-BE: bdnzlrl-                        # encoding: [0x4f,0x00,0x00,0x21]
# CHECK-LE: bdnzlrl-                        # encoding: [0x21,0x00,0x00,0x4f]
            bdnzlrl-

# CHECK-BE: bc 8, 2, target                 # encoding: [0x41,0x02,A,0bAAAAAA00]
# CHECK-LE: bc 8, 2, target                 # encoding: [0bAAAAAA00,A,0x02,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzt 2, target
# CHECK-BE: bca 8, 2, target                # encoding: [0x41,0x02,A,0bAAAAAA10]
# CHECK-LE: bca 8, 2, target                # encoding: [0bAAAAAA10,A,0x02,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzta 2, target
# CHECK-BE: bclr 8, 2, 0                    # encoding: [0x4d,0x02,0x00,0x20]
# CHECK-LE: bclr 8, 2, 0                    # encoding: [0x20,0x00,0x02,0x4d]
            bdnztlr 2
# CHECK-BE: bcl 8, 2, target                # encoding: [0x41,0x02,A,0bAAAAAA01]
# CHECK-LE: bcl 8, 2, target                # encoding: [0bAAAAAA01,A,0x02,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnztl 2, target
# CHECK-BE: bcla 8, 2, target               # encoding: [0x41,0x02,A,0bAAAAAA11]
# CHECK-LE: bcla 8, 2, target               # encoding: [0bAAAAAA11,A,0x02,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnztla 2, target
# CHECK-BE: bclrl 8, 2, 0                   # encoding: [0x4d,0x02,0x00,0x21]
# CHECK-LE: bclrl 8, 2, 0                   # encoding: [0x21,0x00,0x02,0x4d]
            bdnztlrl 2

# CHECK-BE: bc 0, 2, target                 # encoding: [0x40,0x02,A,0bAAAAAA00]
# CHECK-LE: bc 0, 2, target                 # encoding: [0bAAAAAA00,A,0x02,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzf 2, target
# CHECK-BE: bca 0, 2, target                # encoding: [0x40,0x02,A,0bAAAAAA10]
# CHECK-LE: bca 0, 2, target                # encoding: [0bAAAAAA10,A,0x02,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzfa 2, target
# CHECK-BE: bclr 0, 2, 0                    # encoding: [0x4c,0x02,0x00,0x20]
# CHECK-LE: bclr 0, 2, 0                    # encoding: [0x20,0x00,0x02,0x4c]
            bdnzflr 2
# CHECK-BE: bcl 0, 2, target                # encoding: [0x40,0x02,A,0bAAAAAA01]
# CHECK-LE: bcl 0, 2, target                # encoding: [0bAAAAAA01,A,0x02,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzfl 2, target
# CHECK-BE: bcla 0, 2, target               # encoding: [0x40,0x02,A,0bAAAAAA11]
# CHECK-LE: bcla 0, 2, target               # encoding: [0bAAAAAA11,A,0x02,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzfla 2, target
# CHECK-BE: bclrl 0, 2, 0                   # encoding: [0x4c,0x02,0x00,0x21]
# CHECK-LE: bclrl 0, 2, 0                   # encoding: [0x21,0x00,0x02,0x4c]
            bdnzflrl 2

# CHECK-BE: bdz target                      # encoding: [0x42,0x40,A,0bAAAAAA00]
# CHECK-LE: bdz target                      # encoding: [0bAAAAAA00,A,0x40,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdz target
# CHECK-BE: bdza target                     # encoding: [0x42,0x40,A,0bAAAAAA10]
# CHECK-LE: bdza target                     # encoding: [0bAAAAAA10,A,0x40,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdza target
# CHECK-BE: bdzlr                           # encoding: [0x4e,0x40,0x00,0x20]
# CHECK-LE: bdzlr                           # encoding: [0x20,0x00,0x40,0x4e]
            bdzlr
# CHECK-BE: bdzl target                     # encoding: [0x42,0x40,A,0bAAAAAA01]
# CHECK-LE: bdzl target                     # encoding: [0bAAAAAA01,A,0x40,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzl target
# CHECK-BE: bdzla target                    # encoding: [0x42,0x40,A,0bAAAAAA11]
# CHECK-LE: bdzla target                    # encoding: [0bAAAAAA11,A,0x40,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzla target
# CHECK-BE: bdzlrl                          # encoding: [0x4e,0x40,0x00,0x21]
# CHECK-LE: bdzlrl                          # encoding: [0x21,0x00,0x40,0x4e]
            bdzlrl

# CHECK-BE: bdz+ target                     # encoding: [0x43,0x60,A,0bAAAAAA00]
# CHECK-LE: bdz+ target                     # encoding: [0bAAAAAA00,A,0x60,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdz+ target
# CHECK-BE: bdza+ target                    # encoding: [0x43,0x60,A,0bAAAAAA10]
# CHECK-LE: bdza+ target                    # encoding: [0bAAAAAA10,A,0x60,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdza+ target
# CHECK-BE: bdzlr+                          # encoding: [0x4f,0x60,0x00,0x20]
# CHECK-LE: bdzlr+                          # encoding: [0x20,0x00,0x60,0x4f]
            bdzlr+
# CHECK-BE: bdzl+ target                    # encoding: [0x43,0x60,A,0bAAAAAA01]
# CHECK-LE: bdzl+ target                    # encoding: [0bAAAAAA01,A,0x60,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzl+ target
# CHECK-BE: bdzla+ target                   # encoding: [0x43,0x60,A,0bAAAAAA11]
# CHECK-LE: bdzla+ target                   # encoding: [0bAAAAAA11,A,0x60,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzla+ target
# CHECK-BE: bdzlrl+                         # encoding: [0x4f,0x60,0x00,0x21]
# CHECK-LE: bdzlrl+                         # encoding: [0x21,0x00,0x60,0x4f]
            bdzlrl+

# CHECK-BE: bdz- target                     # encoding: [0x43,0x40,A,0bAAAAAA00]
# CHECK-LE: bdz- target                     # encoding: [0bAAAAAA00,A,0x40,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdz- target
# CHECK-BE: bdza- target                    # encoding: [0x43,0x40,A,0bAAAAAA10]
# CHECK-LE: bdza- target                    # encoding: [0bAAAAAA10,A,0x40,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdza- target
# CHECK-BE: bdzlr-                          # encoding: [0x4f,0x40,0x00,0x20]
# CHECK-LE: bdzlr-                          # encoding: [0x20,0x00,0x40,0x4f]
            bdzlr-
# CHECK-BE: bdzl- target                    # encoding: [0x43,0x40,A,0bAAAAAA01]
# CHECK-LE: bdzl- target                    # encoding: [0bAAAAAA01,A,0x40,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzl- target
# CHECK-BE: bdzla- target                   # encoding: [0x43,0x40,A,0bAAAAAA11]
# CHECK-LE: bdzla- target                   # encoding: [0bAAAAAA11,A,0x40,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzla- target
# CHECK-BE: bdzlrl-                         # encoding: [0x4f,0x40,0x00,0x21]
# CHECK-LE: bdzlrl-                         # encoding: [0x21,0x00,0x40,0x4f]
            bdzlrl-

# CHECK-BE: bc 10, 2, target                # encoding: [0x41,0x42,A,0bAAAAAA00]
# CHECK-LE: bc 10, 2, target                # encoding: [0bAAAAAA00,A,0x42,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzt 2, target
# CHECK-BE: bca 10, 2, target               # encoding: [0x41,0x42,A,0bAAAAAA10]
# CHECK-LE: bca 10, 2, target               # encoding: [0bAAAAAA10,A,0x42,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzta 2, target
# CHECK-BE: bclr 10, 2, 0                   # encoding: [0x4d,0x42,0x00,0x20]
# CHECK-LE: bclr 10, 2, 0                   # encoding: [0x20,0x00,0x42,0x4d]
            bdztlr 2
# CHECK-BE: bcl 10, 2, target               # encoding: [0x41,0x42,A,0bAAAAAA01]
# CHECK-LE: bcl 10, 2, target               # encoding: [0bAAAAAA01,A,0x42,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdztl 2, target
# CHECK-BE: bcla 10, 2, target              # encoding: [0x41,0x42,A,0bAAAAAA11]
# CHECK-LE: bcla 10, 2, target              # encoding: [0bAAAAAA11,A,0x42,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdztla 2, target
# CHECK-BE: bclrl 10, 2, 0                  # encoding: [0x4d,0x42,0x00,0x21]
# CHECK-LE: bclrl 10, 2, 0                  # encoding: [0x21,0x00,0x42,0x4d]
            bdztlrl 2

# CHECK-BE: bc 2, 2, target                 # encoding: [0x40,0x42,A,0bAAAAAA00]
# CHECK-LE: bc 2, 2, target                 # encoding: [0bAAAAAA00,A,0x42,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzf 2, target
# CHECK-BE: bca 2, 2, target                # encoding: [0x40,0x42,A,0bAAAAAA10]
# CHECK-LE: bca 2, 2, target                # encoding: [0bAAAAAA10,A,0x42,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzfa 2, target
# CHECK-BE: bclr 2, 2, 0                    # encoding: [0x4c,0x42,0x00,0x20]
# CHECK-LE: bclr 2, 2, 0                    # encoding: [0x20,0x00,0x42,0x4c]
            bdzflr 2
# CHECK-BE: bcl 2, 2, target                # encoding: [0x40,0x42,A,0bAAAAAA01]
# CHECK-LE: bcl 2, 2, target                # encoding: [0bAAAAAA01,A,0x42,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzfl 2, target
# CHECK-BE: bcla 2, 2, target               # encoding: [0x40,0x42,A,0bAAAAAA11]
# CHECK-LE: bcla 2, 2, target               # encoding: [0bAAAAAA11,A,0x42,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzfla 2, target
# CHECK-BE: bclrl 2, 2, 0                   # encoding: [0x4c,0x42,0x00,0x21]
# CHECK-LE: bclrl 2, 2, 0                   # encoding: [0x21,0x00,0x42,0x4c]
            bdzflrl 2

# CHECK-BE: blt 2, target                   # encoding: [0x41,0x88,A,0bAAAAAA00]
# CHECK-LE: blt 2, target                   # encoding: [0bAAAAAA00,A,0x88,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt 2, target
# CHECK-BE: blt 0, target                   # encoding: [0x41,0x80,A,0bAAAAAA00]
# CHECK-LE: blt 0, target                   # encoding: [0bAAAAAA00,A,0x80,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt target
# CHECK-BE: blta 2, target                  # encoding: [0x41,0x88,A,0bAAAAAA10]
# CHECK-LE: blta 2, target                  # encoding: [0bAAAAAA10,A,0x88,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta 2, target
# CHECK-BE: blta 0, target                  # encoding: [0x41,0x80,A,0bAAAAAA10]
# CHECK-LE: blta 0, target                  # encoding: [0bAAAAAA10,A,0x80,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta target
# CHECK-BE: bltlr 2                         # encoding: [0x4d,0x88,0x00,0x20]
# CHECK-LE: bltlr 2                         # encoding: [0x20,0x00,0x88,0x4d]
            bltlr 2
# CHECK-BE: bltlr 0                         # encoding: [0x4d,0x80,0x00,0x20]
# CHECK-LE: bltlr 0                         # encoding: [0x20,0x00,0x80,0x4d]
            bltlr
# CHECK-BE: bltctr 2                        # encoding: [0x4d,0x88,0x04,0x20]
# CHECK-LE: bltctr 2                        # encoding: [0x20,0x04,0x88,0x4d]
            bltctr 2
# CHECK-BE: bltctr 0                        # encoding: [0x4d,0x80,0x04,0x20]
# CHECK-LE: bltctr 0                        # encoding: [0x20,0x04,0x80,0x4d]
            bltctr
# CHECK-BE: bltl 2, target                  # encoding: [0x41,0x88,A,0bAAAAAA01]
# CHECK-LE: bltl 2, target                  # encoding: [0bAAAAAA01,A,0x88,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl 2, target
# CHECK-BE: bltl 0, target                  # encoding: [0x41,0x80,A,0bAAAAAA01]
# CHECK-LE: bltl 0, target                  # encoding: [0bAAAAAA01,A,0x80,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl target
# CHECK-BE: bltla 2, target                 # encoding: [0x41,0x88,A,0bAAAAAA11]
# CHECK-LE: bltla 2, target                 # encoding: [0bAAAAAA11,A,0x88,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla 2, target
# CHECK-BE: bltla 0, target                 # encoding: [0x41,0x80,A,0bAAAAAA11]
# CHECK-LE: bltla 0, target                 # encoding: [0bAAAAAA11,A,0x80,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla target
# CHECK-BE: bltlrl 2                        # encoding: [0x4d,0x88,0x00,0x21]
# CHECK-LE: bltlrl 2                        # encoding: [0x21,0x00,0x88,0x4d]
            bltlrl 2
# CHECK-BE: bltlrl 0                        # encoding: [0x4d,0x80,0x00,0x21]
# CHECK-LE: bltlrl 0                        # encoding: [0x21,0x00,0x80,0x4d]
            bltlrl
# CHECK-BE: bltctrl 2                       # encoding: [0x4d,0x88,0x04,0x21]
# CHECK-LE: bltctrl 2                       # encoding: [0x21,0x04,0x88,0x4d]
            bltctrl 2
# CHECK-BE: bltctrl 0                       # encoding: [0x4d,0x80,0x04,0x21]
# CHECK-LE: bltctrl 0                       # encoding: [0x21,0x04,0x80,0x4d]
            bltctrl

# CHECK-BE: blt+ 2, target                  # encoding: [0x41,0xe8,A,0bAAAAAA00]
# CHECK-LE: blt+ 2, target                  # encoding: [0bAAAAAA00,A,0xe8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt+ 2, target
# CHECK-BE: blt+ 0, target                  # encoding: [0x41,0xe0,A,0bAAAAAA00]
# CHECK-LE: blt+ 0, target                  # encoding: [0bAAAAAA00,A,0xe0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt+ target
# CHECK-BE: blta+ 2, target                 # encoding: [0x41,0xe8,A,0bAAAAAA10]
# CHECK-LE: blta+ 2, target                 # encoding: [0bAAAAAA10,A,0xe8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta+ 2, target
# CHECK-BE: blta+ 0, target                 # encoding: [0x41,0xe0,A,0bAAAAAA10]
# CHECK-LE: blta+ 0, target                 # encoding: [0bAAAAAA10,A,0xe0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta+ target
# CHECK-BE: bltlr+ 2                        # encoding: [0x4d,0xe8,0x00,0x20]
# CHECK-LE: bltlr+ 2                        # encoding: [0x20,0x00,0xe8,0x4d]
            bltlr+ 2
# CHECK-BE: bltlr+ 0                        # encoding: [0x4d,0xe0,0x00,0x20]
# CHECK-LE: bltlr+ 0                        # encoding: [0x20,0x00,0xe0,0x4d]
            bltlr+
# CHECK-BE: bltctr+ 2                       # encoding: [0x4d,0xe8,0x04,0x20]
# CHECK-LE: bltctr+ 2                       # encoding: [0x20,0x04,0xe8,0x4d]
            bltctr+ 2
# CHECK-BE: bltctr+ 0                       # encoding: [0x4d,0xe0,0x04,0x20]
# CHECK-LE: bltctr+ 0                       # encoding: [0x20,0x04,0xe0,0x4d]
            bltctr+
# CHECK-BE: bltl+ 2, target                 # encoding: [0x41,0xe8,A,0bAAAAAA01]
# CHECK-LE: bltl+ 2, target                 # encoding: [0bAAAAAA01,A,0xe8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl+ 2, target
# CHECK-BE: bltl+ 0, target                 # encoding: [0x41,0xe0,A,0bAAAAAA01]
# CHECK-LE: bltl+ 0, target                 # encoding: [0bAAAAAA01,A,0xe0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl+ target
# CHECK-BE: bltla+ 2, target                # encoding: [0x41,0xe8,A,0bAAAAAA11]
# CHECK-LE: bltla+ 2, target                # encoding: [0bAAAAAA11,A,0xe8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla+ 2, target
# CHECK-BE: bltla+ 0, target                # encoding: [0x41,0xe0,A,0bAAAAAA11]
# CHECK-LE: bltla+ 0, target                # encoding: [0bAAAAAA11,A,0xe0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla+ target
# CHECK-BE: bltlrl+ 2                       # encoding: [0x4d,0xe8,0x00,0x21]
# CHECK-LE: bltlrl+ 2                       # encoding: [0x21,0x00,0xe8,0x4d]
            bltlrl+ 2
# CHECK-BE: bltlrl+ 0                       # encoding: [0x4d,0xe0,0x00,0x21]
# CHECK-LE: bltlrl+ 0                       # encoding: [0x21,0x00,0xe0,0x4d]
            bltlrl+
# CHECK-BE: bltctrl+ 2                      # encoding: [0x4d,0xe8,0x04,0x21]
# CHECK-LE: bltctrl+ 2                      # encoding: [0x21,0x04,0xe8,0x4d]
            bltctrl+ 2
# CHECK-BE: bltctrl+ 0                      # encoding: [0x4d,0xe0,0x04,0x21]
# CHECK-LE: bltctrl+ 0                      # encoding: [0x21,0x04,0xe0,0x4d]
            bltctrl+

# CHECK-BE: blt- 2, target                  # encoding: [0x41,0xc8,A,0bAAAAAA00]
# CHECK-LE: blt- 2, target                  # encoding: [0bAAAAAA00,A,0xc8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt- 2, target
# CHECK-BE: blt- 0, target                  # encoding: [0x41,0xc0,A,0bAAAAAA00]
# CHECK-LE: blt- 0, target                  # encoding: [0bAAAAAA00,A,0xc0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt- target
# CHECK-BE: blta- 2, target                 # encoding: [0x41,0xc8,A,0bAAAAAA10]
# CHECK-LE: blta- 2, target                 # encoding: [0bAAAAAA10,A,0xc8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta- 2, target
# CHECK-BE: blta- 0, target                 # encoding: [0x41,0xc0,A,0bAAAAAA10]
# CHECK-LE: blta- 0, target                 # encoding: [0bAAAAAA10,A,0xc0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta- target
# CHECK-BE: bltlr- 2                        # encoding: [0x4d,0xc8,0x00,0x20]
# CHECK-LE: bltlr- 2                        # encoding: [0x20,0x00,0xc8,0x4d]
            bltlr- 2
# CHECK-BE: bltlr- 0                        # encoding: [0x4d,0xc0,0x00,0x20]
# CHECK-LE: bltlr- 0                        # encoding: [0x20,0x00,0xc0,0x4d]
            bltlr-
# CHECK-BE: bltctr- 2                       # encoding: [0x4d,0xc8,0x04,0x20]
# CHECK-LE: bltctr- 2                       # encoding: [0x20,0x04,0xc8,0x4d]
            bltctr- 2
# CHECK-BE: bltctr- 0                       # encoding: [0x4d,0xc0,0x04,0x20]
# CHECK-LE: bltctr- 0                       # encoding: [0x20,0x04,0xc0,0x4d]
            bltctr-
# CHECK-BE: bltl- 2, target                 # encoding: [0x41,0xc8,A,0bAAAAAA01]
# CHECK-LE: bltl- 2, target                 # encoding: [0bAAAAAA01,A,0xc8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl- 2, target
# CHECK-BE: bltl- 0, target                 # encoding: [0x41,0xc0,A,0bAAAAAA01]
# CHECK-LE: bltl- 0, target                 # encoding: [0bAAAAAA01,A,0xc0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl- target
# CHECK-BE: bltla- 2, target                # encoding: [0x41,0xc8,A,0bAAAAAA11]
# CHECK-LE: bltla- 2, target                # encoding: [0bAAAAAA11,A,0xc8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla- 2, target
# CHECK-BE: bltla- 0, target                # encoding: [0x41,0xc0,A,0bAAAAAA11]
# CHECK-LE: bltla- 0, target                # encoding: [0bAAAAAA11,A,0xc0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla- target
# CHECK-BE: bltlrl- 2                       # encoding: [0x4d,0xc8,0x00,0x21]
# CHECK-LE: bltlrl- 2                       # encoding: [0x21,0x00,0xc8,0x4d]
            bltlrl- 2
# CHECK-BE: bltlrl- 0                       # encoding: [0x4d,0xc0,0x00,0x21]
# CHECK-LE: bltlrl- 0                       # encoding: [0x21,0x00,0xc0,0x4d]
            bltlrl-
# CHECK-BE: bltctrl- 2                      # encoding: [0x4d,0xc8,0x04,0x21]
# CHECK-LE: bltctrl- 2                      # encoding: [0x21,0x04,0xc8,0x4d]
            bltctrl- 2
# CHECK-BE: bltctrl- 0                      # encoding: [0x4d,0xc0,0x04,0x21]
# CHECK-LE: bltctrl- 0                      # encoding: [0x21,0x04,0xc0,0x4d]
            bltctrl-

# CHECK-BE: ble 2, target                   # encoding: [0x40,0x89,A,0bAAAAAA00]
# CHECK-LE: ble 2, target                   # encoding: [0bAAAAAA00,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble 2, target
# CHECK-BE: ble 0, target                   # encoding: [0x40,0x81,A,0bAAAAAA00]
# CHECK-LE: ble 0, target                   # encoding: [0bAAAAAA00,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble target
# CHECK-BE: blea 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA10]
# CHECK-LE: blea 2, target                  # encoding: [0bAAAAAA10,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea 2, target
# CHECK-BE: blea 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA10]
# CHECK-LE: blea 0, target                  # encoding: [0bAAAAAA10,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea target
# CHECK-BE: blelr 2                         # encoding: [0x4c,0x89,0x00,0x20]
# CHECK-LE: blelr 2                         # encoding: [0x20,0x00,0x89,0x4c]
            blelr 2
# CHECK-BE: blelr 0                         # encoding: [0x4c,0x81,0x00,0x20]
# CHECK-LE: blelr 0                         # encoding: [0x20,0x00,0x81,0x4c]
            blelr
# CHECK-BE: blectr 2                        # encoding: [0x4c,0x89,0x04,0x20]
# CHECK-LE: blectr 2                        # encoding: [0x20,0x04,0x89,0x4c]
            blectr 2
# CHECK-BE: blectr 0                        # encoding: [0x4c,0x81,0x04,0x20]
# CHECK-LE: blectr 0                        # encoding: [0x20,0x04,0x81,0x4c]
            blectr
# CHECK-BE: blel 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA01]
# CHECK-LE: blel 2, target                  # encoding: [0bAAAAAA01,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel 2, target
# CHECK-BE: blel 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA01]
# CHECK-LE: blel 0, target                  # encoding: [0bAAAAAA01,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel target
# CHECK-BE: blela 2, target                 # encoding: [0x40,0x89,A,0bAAAAAA11]
# CHECK-LE: blela 2, target                 # encoding: [0bAAAAAA11,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela 2, target
# CHECK-BE: blela 0, target                 # encoding: [0x40,0x81,A,0bAAAAAA11]
# CHECK-LE: blela 0, target                 # encoding: [0bAAAAAA11,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela target
# CHECK-BE: blelrl 2                        # encoding: [0x4c,0x89,0x00,0x21]
# CHECK-LE: blelrl 2                        # encoding: [0x21,0x00,0x89,0x4c]
            blelrl 2
# CHECK-BE: blelrl 0                        # encoding: [0x4c,0x81,0x00,0x21]
# CHECK-LE: blelrl 0                        # encoding: [0x21,0x00,0x81,0x4c]
            blelrl
# CHECK-BE: blectrl 2                       # encoding: [0x4c,0x89,0x04,0x21]
# CHECK-LE: blectrl 2                       # encoding: [0x21,0x04,0x89,0x4c]
            blectrl 2
# CHECK-BE: blectrl 0                       # encoding: [0x4c,0x81,0x04,0x21]
# CHECK-LE: blectrl 0                       # encoding: [0x21,0x04,0x81,0x4c]
            blectrl

# CHECK-BE: ble+ 2, target                  # encoding: [0x40,0xe9,A,0bAAAAAA00]
# CHECK-LE: ble+ 2, target                  # encoding: [0bAAAAAA00,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble+ 2, target
# CHECK-BE: ble+ 0, target                  # encoding: [0x40,0xe1,A,0bAAAAAA00]
# CHECK-LE: ble+ 0, target                  # encoding: [0bAAAAAA00,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble+ target
# CHECK-BE: blea+ 2, target                 # encoding: [0x40,0xe9,A,0bAAAAAA10]
# CHECK-LE: blea+ 2, target                 # encoding: [0bAAAAAA10,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea+ 2, target
# CHECK-BE: blea+ 0, target                 # encoding: [0x40,0xe1,A,0bAAAAAA10]
# CHECK-LE: blea+ 0, target                 # encoding: [0bAAAAAA10,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea+ target
# CHECK-BE: blelr+ 2                        # encoding: [0x4c,0xe9,0x00,0x20]
# CHECK-LE: blelr+ 2                        # encoding: [0x20,0x00,0xe9,0x4c]
            blelr+ 2
# CHECK-BE: blelr+ 0                        # encoding: [0x4c,0xe1,0x00,0x20]
# CHECK-LE: blelr+ 0                        # encoding: [0x20,0x00,0xe1,0x4c]
            blelr+
# CHECK-BE: blectr+ 2                       # encoding: [0x4c,0xe9,0x04,0x20]
# CHECK-LE: blectr+ 2                       # encoding: [0x20,0x04,0xe9,0x4c]
            blectr+ 2
# CHECK-BE: blectr+ 0                       # encoding: [0x4c,0xe1,0x04,0x20]
# CHECK-LE: blectr+ 0                       # encoding: [0x20,0x04,0xe1,0x4c]
            blectr+
# CHECK-BE: blel+ 2, target                 # encoding: [0x40,0xe9,A,0bAAAAAA01]
# CHECK-LE: blel+ 2, target                 # encoding: [0bAAAAAA01,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel+ 2, target
# CHECK-BE: blel+ 0, target                 # encoding: [0x40,0xe1,A,0bAAAAAA01]
# CHECK-LE: blel+ 0, target                 # encoding: [0bAAAAAA01,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel+ target
# CHECK-BE: blela+ 2, target                # encoding: [0x40,0xe9,A,0bAAAAAA11]
# CHECK-LE: blela+ 2, target                # encoding: [0bAAAAAA11,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela+ 2, target
# CHECK-BE: blela+ 0, target                # encoding: [0x40,0xe1,A,0bAAAAAA11]
# CHECK-LE: blela+ 0, target                # encoding: [0bAAAAAA11,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela+ target
# CHECK-BE: blelrl+ 2                       # encoding: [0x4c,0xe9,0x00,0x21]
# CHECK-LE: blelrl+ 2                       # encoding: [0x21,0x00,0xe9,0x4c]
            blelrl+ 2
# CHECK-BE: blelrl+ 0                       # encoding: [0x4c,0xe1,0x00,0x21]
# CHECK-LE: blelrl+ 0                       # encoding: [0x21,0x00,0xe1,0x4c]
            blelrl+
# CHECK-BE: blectrl+ 2                      # encoding: [0x4c,0xe9,0x04,0x21]
# CHECK-LE: blectrl+ 2                      # encoding: [0x21,0x04,0xe9,0x4c]
            blectrl+ 2
# CHECK-BE: blectrl+ 0                      # encoding: [0x4c,0xe1,0x04,0x21]
# CHECK-LE: blectrl+ 0                      # encoding: [0x21,0x04,0xe1,0x4c]
            blectrl+

# CHECK-BE: ble- 2, target                  # encoding: [0x40,0xc9,A,0bAAAAAA00]
# CHECK-LE: ble- 2, target                  # encoding: [0bAAAAAA00,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble- 2, target
# CHECK-BE: ble- 0, target                  # encoding: [0x40,0xc1,A,0bAAAAAA00]
# CHECK-LE: ble- 0, target                  # encoding: [0bAAAAAA00,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble- target
# CHECK-BE: blea- 2, target                 # encoding: [0x40,0xc9,A,0bAAAAAA10]
# CHECK-LE: blea- 2, target                 # encoding: [0bAAAAAA10,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea- 2, target
# CHECK-BE: blea- 0, target                 # encoding: [0x40,0xc1,A,0bAAAAAA10]
# CHECK-LE: blea- 0, target                 # encoding: [0bAAAAAA10,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea- target
# CHECK-BE: blelr- 2                        # encoding: [0x4c,0xc9,0x00,0x20]
# CHECK-LE: blelr- 2                        # encoding: [0x20,0x00,0xc9,0x4c]
            blelr- 2
# CHECK-BE: blelr- 0                        # encoding: [0x4c,0xc1,0x00,0x20]
# CHECK-LE: blelr- 0                        # encoding: [0x20,0x00,0xc1,0x4c]
            blelr-
# CHECK-BE: blectr- 2                       # encoding: [0x4c,0xc9,0x04,0x20]
# CHECK-LE: blectr- 2                       # encoding: [0x20,0x04,0xc9,0x4c]
            blectr- 2
# CHECK-BE: blectr- 0                       # encoding: [0x4c,0xc1,0x04,0x20]
# CHECK-LE: blectr- 0                       # encoding: [0x20,0x04,0xc1,0x4c]
            blectr-
# CHECK-BE: blel- 2, target                 # encoding: [0x40,0xc9,A,0bAAAAAA01]
# CHECK-LE: blel- 2, target                 # encoding: [0bAAAAAA01,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel- 2, target
# CHECK-BE: blel- 0, target                 # encoding: [0x40,0xc1,A,0bAAAAAA01]
# CHECK-LE: blel- 0, target                 # encoding: [0bAAAAAA01,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel- target
# CHECK-BE: blela- 2, target                # encoding: [0x40,0xc9,A,0bAAAAAA11]
# CHECK-LE: blela- 2, target                # encoding: [0bAAAAAA11,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela- 2, target
# CHECK-BE: blela- 0, target                # encoding: [0x40,0xc1,A,0bAAAAAA11]
# CHECK-LE: blela- 0, target                # encoding: [0bAAAAAA11,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela- target
# CHECK-BE: blelrl- 2                       # encoding: [0x4c,0xc9,0x00,0x21]
# CHECK-LE: blelrl- 2                       # encoding: [0x21,0x00,0xc9,0x4c]
            blelrl- 2
# CHECK-BE: blelrl- 0                       # encoding: [0x4c,0xc1,0x00,0x21]
# CHECK-LE: blelrl- 0                       # encoding: [0x21,0x00,0xc1,0x4c]
            blelrl-
# CHECK-BE: blectrl- 2                      # encoding: [0x4c,0xc9,0x04,0x21]
# CHECK-LE: blectrl- 2                      # encoding: [0x21,0x04,0xc9,0x4c]
            blectrl- 2
# CHECK-BE: blectrl- 0                      # encoding: [0x4c,0xc1,0x04,0x21]
# CHECK-LE: blectrl- 0                      # encoding: [0x21,0x04,0xc1,0x4c]
            blectrl-

# CHECK-BE: beq 2, target                   # encoding: [0x41,0x8a,A,0bAAAAAA00]
# CHECK-LE: beq 2, target                   # encoding: [0bAAAAAA00,A,0x8a,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq 2, target
# CHECK-BE: beq 0, target                   # encoding: [0x41,0x82,A,0bAAAAAA00]
# CHECK-LE: beq 0, target                   # encoding: [0bAAAAAA00,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq target
# CHECK-BE: beqa 2, target                  # encoding: [0x41,0x8a,A,0bAAAAAA10]
# CHECK-LE: beqa 2, target                  # encoding: [0bAAAAAA10,A,0x8a,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa 2, target
# CHECK-BE: beqa 0, target                  # encoding: [0x41,0x82,A,0bAAAAAA10]
# CHECK-LE: beqa 0, target                  # encoding: [0bAAAAAA10,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa target
# CHECK-BE: beqlr 2                         # encoding: [0x4d,0x8a,0x00,0x20]
# CHECK-LE: beqlr 2                         # encoding: [0x20,0x00,0x8a,0x4d]
            beqlr 2
# CHECK-BE: beqlr 0                         # encoding: [0x4d,0x82,0x00,0x20]
# CHECK-LE: beqlr 0                         # encoding: [0x20,0x00,0x82,0x4d]
            beqlr
# CHECK-BE: beqctr 2                        # encoding: [0x4d,0x8a,0x04,0x20]
# CHECK-LE: beqctr 2                        # encoding: [0x20,0x04,0x8a,0x4d]
            beqctr 2
# CHECK-BE: beqctr 0                        # encoding: [0x4d,0x82,0x04,0x20]
# CHECK-LE: beqctr 0                        # encoding: [0x20,0x04,0x82,0x4d]
            beqctr
# CHECK-BE: beql 2, target                  # encoding: [0x41,0x8a,A,0bAAAAAA01]
# CHECK-LE: beql 2, target                  # encoding: [0bAAAAAA01,A,0x8a,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql 2, target
# CHECK-BE: beql 0, target                  # encoding: [0x41,0x82,A,0bAAAAAA01]
# CHECK-LE: beql 0, target                  # encoding: [0bAAAAAA01,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql target
# CHECK-BE: beqla 2, target                 # encoding: [0x41,0x8a,A,0bAAAAAA11]
# CHECK-LE: beqla 2, target                 # encoding: [0bAAAAAA11,A,0x8a,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla 2, target
# CHECK-BE: beqla 0, target                 # encoding: [0x41,0x82,A,0bAAAAAA11]
# CHECK-LE: beqla 0, target                 # encoding: [0bAAAAAA11,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla target
# CHECK-BE: beqlrl 2                        # encoding: [0x4d,0x8a,0x00,0x21]
# CHECK-LE: beqlrl 2                        # encoding: [0x21,0x00,0x8a,0x4d]
            beqlrl 2
# CHECK-BE: beqlrl 0                        # encoding: [0x4d,0x82,0x00,0x21]
# CHECK-LE: beqlrl 0                        # encoding: [0x21,0x00,0x82,0x4d]
            beqlrl
# CHECK-BE: beqctrl 2                       # encoding: [0x4d,0x8a,0x04,0x21]
# CHECK-LE: beqctrl 2                       # encoding: [0x21,0x04,0x8a,0x4d]
            beqctrl 2
# CHECK-BE: beqctrl 0                       # encoding: [0x4d,0x82,0x04,0x21]
# CHECK-LE: beqctrl 0                       # encoding: [0x21,0x04,0x82,0x4d]
            beqctrl

# CHECK-BE: beq+ 2, target                  # encoding: [0x41,0xea,A,0bAAAAAA00]
# CHECK-LE: beq+ 2, target                  # encoding: [0bAAAAAA00,A,0xea,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq+ 2, target
# CHECK-BE: beq+ 0, target                  # encoding: [0x41,0xe2,A,0bAAAAAA00]
# CHECK-LE: beq+ 0, target                  # encoding: [0bAAAAAA00,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq+ target
# CHECK-BE: beqa+ 2, target                 # encoding: [0x41,0xea,A,0bAAAAAA10]
# CHECK-LE: beqa+ 2, target                 # encoding: [0bAAAAAA10,A,0xea,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa+ 2, target
# CHECK-BE: beqa+ 0, target                 # encoding: [0x41,0xe2,A,0bAAAAAA10]
# CHECK-LE: beqa+ 0, target                 # encoding: [0bAAAAAA10,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa+ target
# CHECK-BE: beqlr+ 2                        # encoding: [0x4d,0xea,0x00,0x20]
# CHECK-LE: beqlr+ 2                        # encoding: [0x20,0x00,0xea,0x4d]
            beqlr+ 2
# CHECK-BE: beqlr+ 0                        # encoding: [0x4d,0xe2,0x00,0x20]
# CHECK-LE: beqlr+ 0                        # encoding: [0x20,0x00,0xe2,0x4d]
            beqlr+
# CHECK-BE: beqctr+ 2                       # encoding: [0x4d,0xea,0x04,0x20]
# CHECK-LE: beqctr+ 2                       # encoding: [0x20,0x04,0xea,0x4d]
            beqctr+ 2
# CHECK-BE: beqctr+ 0                       # encoding: [0x4d,0xe2,0x04,0x20]
# CHECK-LE: beqctr+ 0                       # encoding: [0x20,0x04,0xe2,0x4d]
            beqctr+
# CHECK-BE: beql+ 2, target                 # encoding: [0x41,0xea,A,0bAAAAAA01]
# CHECK-LE: beql+ 2, target                 # encoding: [0bAAAAAA01,A,0xea,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql+ 2, target
# CHECK-BE: beql+ 0, target                 # encoding: [0x41,0xe2,A,0bAAAAAA01]
# CHECK-LE: beql+ 0, target                 # encoding: [0bAAAAAA01,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql+ target
# CHECK-BE: beqla+ 2, target                # encoding: [0x41,0xea,A,0bAAAAAA11]
# CHECK-LE: beqla+ 2, target                # encoding: [0bAAAAAA11,A,0xea,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla+ 2, target
# CHECK-BE: beqla+ 0, target                # encoding: [0x41,0xe2,A,0bAAAAAA11]
# CHECK-LE: beqla+ 0, target                # encoding: [0bAAAAAA11,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla+ target
# CHECK-BE: beqlrl+ 2                       # encoding: [0x4d,0xea,0x00,0x21]
# CHECK-LE: beqlrl+ 2                       # encoding: [0x21,0x00,0xea,0x4d]
            beqlrl+ 2
# CHECK-BE: beqlrl+ 0                       # encoding: [0x4d,0xe2,0x00,0x21]
# CHECK-LE: beqlrl+ 0                       # encoding: [0x21,0x00,0xe2,0x4d]
            beqlrl+
# CHECK-BE: beqctrl+ 2                      # encoding: [0x4d,0xea,0x04,0x21]
# CHECK-LE: beqctrl+ 2                      # encoding: [0x21,0x04,0xea,0x4d]
            beqctrl+ 2
# CHECK-BE: beqctrl+ 0                      # encoding: [0x4d,0xe2,0x04,0x21]
# CHECK-LE: beqctrl+ 0                      # encoding: [0x21,0x04,0xe2,0x4d]
            beqctrl+

# CHECK-BE: beq- 2, target                  # encoding: [0x41,0xca,A,0bAAAAAA00]
# CHECK-LE: beq- 2, target                  # encoding: [0bAAAAAA00,A,0xca,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq- 2, target
# CHECK-BE: beq- 0, target                  # encoding: [0x41,0xc2,A,0bAAAAAA00]
# CHECK-LE: beq- 0, target                  # encoding: [0bAAAAAA00,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq- target
# CHECK-BE: beqa- 2, target                 # encoding: [0x41,0xca,A,0bAAAAAA10]
# CHECK-LE: beqa- 2, target                 # encoding: [0bAAAAAA10,A,0xca,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa- 2, target
# CHECK-BE: beqa- 0, target                 # encoding: [0x41,0xc2,A,0bAAAAAA10]
# CHECK-LE: beqa- 0, target                 # encoding: [0bAAAAAA10,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa- target
# CHECK-BE: beqlr- 2                        # encoding: [0x4d,0xca,0x00,0x20]
# CHECK-LE: beqlr- 2                        # encoding: [0x20,0x00,0xca,0x4d]
            beqlr- 2
# CHECK-BE: beqlr- 0                        # encoding: [0x4d,0xc2,0x00,0x20]
# CHECK-LE: beqlr- 0                        # encoding: [0x20,0x00,0xc2,0x4d]
            beqlr-
# CHECK-BE: beqctr- 2                       # encoding: [0x4d,0xca,0x04,0x20]
# CHECK-LE: beqctr- 2                       # encoding: [0x20,0x04,0xca,0x4d]
            beqctr- 2
# CHECK-BE: beqctr- 0                       # encoding: [0x4d,0xc2,0x04,0x20]
# CHECK-LE: beqctr- 0                       # encoding: [0x20,0x04,0xc2,0x4d]
            beqctr-
# CHECK-BE: beql- 2, target                 # encoding: [0x41,0xca,A,0bAAAAAA01]
# CHECK-LE: beql- 2, target                 # encoding: [0bAAAAAA01,A,0xca,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql- 2, target
# CHECK-BE: beql- 0, target                 # encoding: [0x41,0xc2,A,0bAAAAAA01]
# CHECK-LE: beql- 0, target                 # encoding: [0bAAAAAA01,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql- target
# CHECK-BE: beqla- 2, target                # encoding: [0x41,0xca,A,0bAAAAAA11]
# CHECK-LE: beqla- 2, target                # encoding: [0bAAAAAA11,A,0xca,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla- 2, target
# CHECK-BE: beqla- 0, target                # encoding: [0x41,0xc2,A,0bAAAAAA11]
# CHECK-LE: beqla- 0, target                # encoding: [0bAAAAAA11,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla- target
# CHECK-BE: beqlrl- 2                       # encoding: [0x4d,0xca,0x00,0x21]
# CHECK-LE: beqlrl- 2                       # encoding: [0x21,0x00,0xca,0x4d]
            beqlrl- 2
# CHECK-BE: beqlrl- 0                       # encoding: [0x4d,0xc2,0x00,0x21]
# CHECK-LE: beqlrl- 0                       # encoding: [0x21,0x00,0xc2,0x4d]
            beqlrl-
# CHECK-BE: beqctrl- 2                      # encoding: [0x4d,0xca,0x04,0x21]
# CHECK-LE: beqctrl- 2                      # encoding: [0x21,0x04,0xca,0x4d]
            beqctrl- 2
# CHECK-BE: beqctrl- 0                      # encoding: [0x4d,0xc2,0x04,0x21]
# CHECK-LE: beqctrl- 0                      # encoding: [0x21,0x04,0xc2,0x4d]
            beqctrl-

# CHECK-BE: bge 2, target                   # encoding: [0x40,0x88,A,0bAAAAAA00]
# CHECK-LE: bge 2, target                   # encoding: [0bAAAAAA00,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge 2, target
# CHECK-BE: bge 0, target                   # encoding: [0x40,0x80,A,0bAAAAAA00]
# CHECK-LE: bge 0, target                   # encoding: [0bAAAAAA00,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge target
# CHECK-BE: bgea 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA10]
# CHECK-LE: bgea 2, target                  # encoding: [0bAAAAAA10,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea 2, target
# CHECK-BE: bgea 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA10]
# CHECK-LE: bgea 0, target                  # encoding: [0bAAAAAA10,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea target
# CHECK-BE: bgelr 2                         # encoding: [0x4c,0x88,0x00,0x20]
# CHECK-LE: bgelr 2                         # encoding: [0x20,0x00,0x88,0x4c]
            bgelr 2
# CHECK-BE: bgelr 0                         # encoding: [0x4c,0x80,0x00,0x20]
# CHECK-LE: bgelr 0                         # encoding: [0x20,0x00,0x80,0x4c]
            bgelr
# CHECK-BE: bgectr 2                        # encoding: [0x4c,0x88,0x04,0x20]
# CHECK-LE: bgectr 2                        # encoding: [0x20,0x04,0x88,0x4c]
            bgectr 2
# CHECK-BE: bgectr 0                        # encoding: [0x4c,0x80,0x04,0x20]
# CHECK-LE: bgectr 0                        # encoding: [0x20,0x04,0x80,0x4c]
            bgectr
# CHECK-BE: bgel 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA01]
# CHECK-LE: bgel 2, target                  # encoding: [0bAAAAAA01,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel 2, target
# CHECK-BE: bgel 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA01]
# CHECK-LE: bgel 0, target                  # encoding: [0bAAAAAA01,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel target
# CHECK-BE: bgela 2, target                 # encoding: [0x40,0x88,A,0bAAAAAA11]
# CHECK-LE: bgela 2, target                 # encoding: [0bAAAAAA11,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela 2, target
# CHECK-BE: bgela 0, target                 # encoding: [0x40,0x80,A,0bAAAAAA11]
# CHECK-LE: bgela 0, target                 # encoding: [0bAAAAAA11,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela target
# CHECK-BE: bgelrl 2                        # encoding: [0x4c,0x88,0x00,0x21]
# CHECK-LE: bgelrl 2                        # encoding: [0x21,0x00,0x88,0x4c]
            bgelrl 2
# CHECK-BE: bgelrl 0                        # encoding: [0x4c,0x80,0x00,0x21]
# CHECK-LE: bgelrl 0                        # encoding: [0x21,0x00,0x80,0x4c]
            bgelrl
# CHECK-BE: bgectrl 2                       # encoding: [0x4c,0x88,0x04,0x21]
# CHECK-LE: bgectrl 2                       # encoding: [0x21,0x04,0x88,0x4c]
            bgectrl 2
# CHECK-BE: bgectrl 0                       # encoding: [0x4c,0x80,0x04,0x21]
# CHECK-LE: bgectrl 0                       # encoding: [0x21,0x04,0x80,0x4c]
            bgectrl

# CHECK-BE: bge+ 2, target                   # encoding: [0x40,0xe8,A,0bAAAAAA00]
# CHECK-LE: bge+ 2, target                   # encoding: [0bAAAAAA00,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge+ 2, target
# CHECK-BE: bge+ 0, target                   # encoding: [0x40,0xe0,A,0bAAAAAA00]
# CHECK-LE: bge+ 0, target                   # encoding: [0bAAAAAA00,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge+ target
# CHECK-BE: bgea+ 2, target                  # encoding: [0x40,0xe8,A,0bAAAAAA10]
# CHECK-LE: bgea+ 2, target                  # encoding: [0bAAAAAA10,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea+ 2, target
# CHECK-BE: bgea+ 0, target                  # encoding: [0x40,0xe0,A,0bAAAAAA10]
# CHECK-LE: bgea+ 0, target                  # encoding: [0bAAAAAA10,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea+ target
# CHECK-BE: bgelr+ 2                         # encoding: [0x4c,0xe8,0x00,0x20]
# CHECK-LE: bgelr+ 2                         # encoding: [0x20,0x00,0xe8,0x4c]
            bgelr+ 2
# CHECK-BE: bgelr+ 0                         # encoding: [0x4c,0xe0,0x00,0x20]
# CHECK-LE: bgelr+ 0                         # encoding: [0x20,0x00,0xe0,0x4c]
            bgelr+
# CHECK-BE: bgectr+ 2                        # encoding: [0x4c,0xe8,0x04,0x20]
# CHECK-LE: bgectr+ 2                        # encoding: [0x20,0x04,0xe8,0x4c]
            bgectr+ 2
# CHECK-BE: bgectr+ 0                        # encoding: [0x4c,0xe0,0x04,0x20]
# CHECK-LE: bgectr+ 0                        # encoding: [0x20,0x04,0xe0,0x4c]
            bgectr+
# CHECK-BE: bgel+ 2, target                  # encoding: [0x40,0xe8,A,0bAAAAAA01]
# CHECK-LE: bgel+ 2, target                  # encoding: [0bAAAAAA01,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel+ 2, target
# CHECK-BE: bgel+ 0, target                  # encoding: [0x40,0xe0,A,0bAAAAAA01]
# CHECK-LE: bgel+ 0, target                  # encoding: [0bAAAAAA01,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel+ target
# CHECK-BE: bgela+ 2, target                 # encoding: [0x40,0xe8,A,0bAAAAAA11]
# CHECK-LE: bgela+ 2, target                 # encoding: [0bAAAAAA11,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela+ 2, target
# CHECK-BE: bgela+ 0, target                 # encoding: [0x40,0xe0,A,0bAAAAAA11]
# CHECK-LE: bgela+ 0, target                 # encoding: [0bAAAAAA11,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela+ target
# CHECK-BE: bgelrl+ 2                        # encoding: [0x4c,0xe8,0x00,0x21]
# CHECK-LE: bgelrl+ 2                        # encoding: [0x21,0x00,0xe8,0x4c]
            bgelrl+ 2
# CHECK-BE: bgelrl+ 0                        # encoding: [0x4c,0xe0,0x00,0x21]
# CHECK-LE: bgelrl+ 0                        # encoding: [0x21,0x00,0xe0,0x4c]
            bgelrl+
# CHECK-BE: bgectrl+ 2                       # encoding: [0x4c,0xe8,0x04,0x21]
# CHECK-LE: bgectrl+ 2                       # encoding: [0x21,0x04,0xe8,0x4c]
            bgectrl+ 2
# CHECK-BE: bgectrl+ 0                       # encoding: [0x4c,0xe0,0x04,0x21]
# CHECK-LE: bgectrl+ 0                       # encoding: [0x21,0x04,0xe0,0x4c]
            bgectrl+

# CHECK-BE: bge- 2, target                   # encoding: [0x40,0xc8,A,0bAAAAAA00]
# CHECK-LE: bge- 2, target                   # encoding: [0bAAAAAA00,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge- 2, target
# CHECK-BE: bge- 0, target                   # encoding: [0x40,0xc0,A,0bAAAAAA00]
# CHECK-LE: bge- 0, target                   # encoding: [0bAAAAAA00,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge- target
# CHECK-BE: bgea- 2, target                  # encoding: [0x40,0xc8,A,0bAAAAAA10]
# CHECK-LE: bgea- 2, target                  # encoding: [0bAAAAAA10,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea- 2, target
# CHECK-BE: bgea- 0, target                  # encoding: [0x40,0xc0,A,0bAAAAAA10]
# CHECK-LE: bgea- 0, target                  # encoding: [0bAAAAAA10,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea- target
# CHECK-BE: bgelr- 2                         # encoding: [0x4c,0xc8,0x00,0x20]
# CHECK-LE: bgelr- 2                         # encoding: [0x20,0x00,0xc8,0x4c]
            bgelr- 2
# CHECK-BE: bgelr- 0                         # encoding: [0x4c,0xc0,0x00,0x20]
# CHECK-LE: bgelr- 0                         # encoding: [0x20,0x00,0xc0,0x4c]
            bgelr-
# CHECK-BE: bgectr- 2                        # encoding: [0x4c,0xc8,0x04,0x20]
# CHECK-LE: bgectr- 2                        # encoding: [0x20,0x04,0xc8,0x4c]
            bgectr- 2
# CHECK-BE: bgectr- 0                        # encoding: [0x4c,0xc0,0x04,0x20]
# CHECK-LE: bgectr- 0                        # encoding: [0x20,0x04,0xc0,0x4c]
            bgectr-
# CHECK-BE: bgel- 2, target                  # encoding: [0x40,0xc8,A,0bAAAAAA01]
# CHECK-LE: bgel- 2, target                  # encoding: [0bAAAAAA01,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel- 2, target
# CHECK-BE: bgel- 0, target                  # encoding: [0x40,0xc0,A,0bAAAAAA01]
# CHECK-LE: bgel- 0, target                  # encoding: [0bAAAAAA01,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel- target
# CHECK-BE: bgela- 2, target                 # encoding: [0x40,0xc8,A,0bAAAAAA11]
# CHECK-LE: bgela- 2, target                 # encoding: [0bAAAAAA11,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela- 2, target
# CHECK-BE: bgela- 0, target                 # encoding: [0x40,0xc0,A,0bAAAAAA11]
# CHECK-LE: bgela- 0, target                 # encoding: [0bAAAAAA11,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela- target
# CHECK-BE: bgelrl- 2                        # encoding: [0x4c,0xc8,0x00,0x21]
# CHECK-LE: bgelrl- 2                        # encoding: [0x21,0x00,0xc8,0x4c]
            bgelrl- 2
# CHECK-BE: bgelrl- 0                        # encoding: [0x4c,0xc0,0x00,0x21]
# CHECK-LE: bgelrl- 0                        # encoding: [0x21,0x00,0xc0,0x4c]
            bgelrl-
# CHECK-BE: bgectrl- 2                       # encoding: [0x4c,0xc8,0x04,0x21]
# CHECK-LE: bgectrl- 2                       # encoding: [0x21,0x04,0xc8,0x4c]
            bgectrl- 2
# CHECK-BE: bgectrl- 0                       # encoding: [0x4c,0xc0,0x04,0x21]
# CHECK-LE: bgectrl- 0                       # encoding: [0x21,0x04,0xc0,0x4c]
            bgectrl-

# CHECK-BE: bgt 2, target                   # encoding: [0x41,0x89,A,0bAAAAAA00]
# CHECK-LE: bgt 2, target                   # encoding: [0bAAAAAA00,A,0x89,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt 2, target
# CHECK-BE: bgt 0, target                   # encoding: [0x41,0x81,A,0bAAAAAA00]
# CHECK-LE: bgt 0, target                   # encoding: [0bAAAAAA00,A,0x81,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt target
# CHECK-BE: bgta 2, target                  # encoding: [0x41,0x89,A,0bAAAAAA10]
# CHECK-LE: bgta 2, target                  # encoding: [0bAAAAAA10,A,0x89,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta 2, target
# CHECK-BE: bgta 0, target                  # encoding: [0x41,0x81,A,0bAAAAAA10]
# CHECK-LE: bgta 0, target                  # encoding: [0bAAAAAA10,A,0x81,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta target
# CHECK-BE: bgtlr 2                         # encoding: [0x4d,0x89,0x00,0x20]
# CHECK-LE: bgtlr 2                         # encoding: [0x20,0x00,0x89,0x4d]
            bgtlr 2
# CHECK-BE: bgtlr 0                         # encoding: [0x4d,0x81,0x00,0x20]
# CHECK-LE: bgtlr 0                         # encoding: [0x20,0x00,0x81,0x4d]
            bgtlr
# CHECK-BE: bgtctr 2                        # encoding: [0x4d,0x89,0x04,0x20]
# CHECK-LE: bgtctr 2                        # encoding: [0x20,0x04,0x89,0x4d]
            bgtctr 2
# CHECK-BE: bgtctr 0                        # encoding: [0x4d,0x81,0x04,0x20]
# CHECK-LE: bgtctr 0                        # encoding: [0x20,0x04,0x81,0x4d]
            bgtctr
# CHECK-BE: bgtl 2, target                  # encoding: [0x41,0x89,A,0bAAAAAA01]
# CHECK-LE: bgtl 2, target                  # encoding: [0bAAAAAA01,A,0x89,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl 2, target
# CHECK-BE: bgtl 0, target                  # encoding: [0x41,0x81,A,0bAAAAAA01]
# CHECK-LE: bgtl 0, target                  # encoding: [0bAAAAAA01,A,0x81,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl target
# CHECK-BE: bgtla 2, target                 # encoding: [0x41,0x89,A,0bAAAAAA11]
# CHECK-LE: bgtla 2, target                 # encoding: [0bAAAAAA11,A,0x89,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla 2, target
# CHECK-BE: bgtla 0, target                 # encoding: [0x41,0x81,A,0bAAAAAA11]
# CHECK-LE: bgtla 0, target                 # encoding: [0bAAAAAA11,A,0x81,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla target
# CHECK-BE: bgtlrl 2                        # encoding: [0x4d,0x89,0x00,0x21]
# CHECK-LE: bgtlrl 2                        # encoding: [0x21,0x00,0x89,0x4d]
            bgtlrl 2
# CHECK-BE: bgtlrl 0                        # encoding: [0x4d,0x81,0x00,0x21]
# CHECK-LE: bgtlrl 0                        # encoding: [0x21,0x00,0x81,0x4d]
            bgtlrl
# CHECK-BE: bgtctrl 2                       # encoding: [0x4d,0x89,0x04,0x21]
# CHECK-LE: bgtctrl 2                       # encoding: [0x21,0x04,0x89,0x4d]
            bgtctrl 2
# CHECK-BE: bgtctrl 0                       # encoding: [0x4d,0x81,0x04,0x21]
# CHECK-LE: bgtctrl 0                       # encoding: [0x21,0x04,0x81,0x4d]
            bgtctrl

# CHECK-BE: bgt+ 2, target                  # encoding: [0x41,0xe9,A,0bAAAAAA00]
# CHECK-LE: bgt+ 2, target                  # encoding: [0bAAAAAA00,A,0xe9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt+ 2, target
# CHECK-BE: bgt+ 0, target                  # encoding: [0x41,0xe1,A,0bAAAAAA00]
# CHECK-LE: bgt+ 0, target                  # encoding: [0bAAAAAA00,A,0xe1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt+ target
# CHECK-BE: bgta+ 2, target                 # encoding: [0x41,0xe9,A,0bAAAAAA10]
# CHECK-LE: bgta+ 2, target                 # encoding: [0bAAAAAA10,A,0xe9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta+ 2, target
# CHECK-BE: bgta+ 0, target                 # encoding: [0x41,0xe1,A,0bAAAAAA10]
# CHECK-LE: bgta+ 0, target                 # encoding: [0bAAAAAA10,A,0xe1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta+ target
# CHECK-BE: bgtlr+ 2                        # encoding: [0x4d,0xe9,0x00,0x20]
# CHECK-LE: bgtlr+ 2                        # encoding: [0x20,0x00,0xe9,0x4d]
            bgtlr+ 2
# CHECK-BE: bgtlr+ 0                        # encoding: [0x4d,0xe1,0x00,0x20]
# CHECK-LE: bgtlr+ 0                        # encoding: [0x20,0x00,0xe1,0x4d]
            bgtlr+
# CHECK-BE: bgtctr+ 2                       # encoding: [0x4d,0xe9,0x04,0x20]
# CHECK-LE: bgtctr+ 2                       # encoding: [0x20,0x04,0xe9,0x4d]
            bgtctr+ 2
# CHECK-BE: bgtctr+ 0                       # encoding: [0x4d,0xe1,0x04,0x20]
# CHECK-LE: bgtctr+ 0                       # encoding: [0x20,0x04,0xe1,0x4d]
            bgtctr+
# CHECK-BE: bgtl+ 2, target                 # encoding: [0x41,0xe9,A,0bAAAAAA01]
# CHECK-LE: bgtl+ 2, target                 # encoding: [0bAAAAAA01,A,0xe9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl+ 2, target
# CHECK-BE: bgtl+ 0, target                 # encoding: [0x41,0xe1,A,0bAAAAAA01]
# CHECK-LE: bgtl+ 0, target                 # encoding: [0bAAAAAA01,A,0xe1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl+ target
# CHECK-BE: bgtla+ 2, target                # encoding: [0x41,0xe9,A,0bAAAAAA11]
# CHECK-LE: bgtla+ 2, target                # encoding: [0bAAAAAA11,A,0xe9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla+ 2, target
# CHECK-BE: bgtla+ 0, target                # encoding: [0x41,0xe1,A,0bAAAAAA11]
# CHECK-LE: bgtla+ 0, target                # encoding: [0bAAAAAA11,A,0xe1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla+ target
# CHECK-BE: bgtlrl+ 2                       # encoding: [0x4d,0xe9,0x00,0x21]
# CHECK-LE: bgtlrl+ 2                       # encoding: [0x21,0x00,0xe9,0x4d]
            bgtlrl+ 2
# CHECK-BE: bgtlrl+ 0                       # encoding: [0x4d,0xe1,0x00,0x21]
# CHECK-LE: bgtlrl+ 0                       # encoding: [0x21,0x00,0xe1,0x4d]
            bgtlrl+
# CHECK-BE: bgtctrl+ 2                      # encoding: [0x4d,0xe9,0x04,0x21]
# CHECK-LE: bgtctrl+ 2                      # encoding: [0x21,0x04,0xe9,0x4d]
            bgtctrl+ 2
# CHECK-BE: bgtctrl+ 0                      # encoding: [0x4d,0xe1,0x04,0x21]
# CHECK-LE: bgtctrl+ 0                      # encoding: [0x21,0x04,0xe1,0x4d]
            bgtctrl+

# CHECK-BE: bgt- 2, target                  # encoding: [0x41,0xc9,A,0bAAAAAA00]
# CHECK-LE: bgt- 2, target                  # encoding: [0bAAAAAA00,A,0xc9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt- 2, target
# CHECK-BE: bgt- 0, target                  # encoding: [0x41,0xc1,A,0bAAAAAA00]
# CHECK-LE: bgt- 0, target                  # encoding: [0bAAAAAA00,A,0xc1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt- target
# CHECK-BE: bgta- 2, target                 # encoding: [0x41,0xc9,A,0bAAAAAA10]
# CHECK-LE: bgta- 2, target                 # encoding: [0bAAAAAA10,A,0xc9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta- 2, target
# CHECK-BE: bgta- 0, target                 # encoding: [0x41,0xc1,A,0bAAAAAA10]
# CHECK-LE: bgta- 0, target                 # encoding: [0bAAAAAA10,A,0xc1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta- target
# CHECK-BE: bgtlr- 2                        # encoding: [0x4d,0xc9,0x00,0x20]
# CHECK-LE: bgtlr- 2                        # encoding: [0x20,0x00,0xc9,0x4d]
            bgtlr- 2
# CHECK-BE: bgtlr- 0                        # encoding: [0x4d,0xc1,0x00,0x20]
# CHECK-LE: bgtlr- 0                        # encoding: [0x20,0x00,0xc1,0x4d]
            bgtlr-
# CHECK-BE: bgtctr- 2                       # encoding: [0x4d,0xc9,0x04,0x20]
# CHECK-LE: bgtctr- 2                       # encoding: [0x20,0x04,0xc9,0x4d]
            bgtctr- 2
# CHECK-BE: bgtctr- 0                       # encoding: [0x4d,0xc1,0x04,0x20]
# CHECK-LE: bgtctr- 0                       # encoding: [0x20,0x04,0xc1,0x4d]
            bgtctr-
# CHECK-BE: bgtl- 2, target                 # encoding: [0x41,0xc9,A,0bAAAAAA01]
# CHECK-LE: bgtl- 2, target                 # encoding: [0bAAAAAA01,A,0xc9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl- 2, target
# CHECK-BE: bgtl- 0, target                 # encoding: [0x41,0xc1,A,0bAAAAAA01]
# CHECK-LE: bgtl- 0, target                 # encoding: [0bAAAAAA01,A,0xc1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl- target
# CHECK-BE: bgtla- 2, target                # encoding: [0x41,0xc9,A,0bAAAAAA11]
# CHECK-LE: bgtla- 2, target                # encoding: [0bAAAAAA11,A,0xc9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla- 2, target
# CHECK-BE: bgtla- 0, target                # encoding: [0x41,0xc1,A,0bAAAAAA11]
# CHECK-LE: bgtla- 0, target                # encoding: [0bAAAAAA11,A,0xc1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla- target
# CHECK-BE: bgtlrl- 2                       # encoding: [0x4d,0xc9,0x00,0x21]
# CHECK-LE: bgtlrl- 2                       # encoding: [0x21,0x00,0xc9,0x4d]
            bgtlrl- 2
# CHECK-BE: bgtlrl- 0                       # encoding: [0x4d,0xc1,0x00,0x21]
# CHECK-LE: bgtlrl- 0                       # encoding: [0x21,0x00,0xc1,0x4d]
            bgtlrl-
# CHECK-BE: bgtctrl- 2                      # encoding: [0x4d,0xc9,0x04,0x21]
# CHECK-LE: bgtctrl- 2                      # encoding: [0x21,0x04,0xc9,0x4d]
            bgtctrl- 2
# CHECK-BE: bgtctrl- 0                      # encoding: [0x4d,0xc1,0x04,0x21]
# CHECK-LE: bgtctrl- 0                      # encoding: [0x21,0x04,0xc1,0x4d]
            bgtctrl-

# CHECK-BE: bge 2, target                   # encoding: [0x40,0x88,A,0bAAAAAA00]
# CHECK-LE: bge 2, target                   # encoding: [0bAAAAAA00,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl 2, target
# CHECK-BE: bge 0, target                   # encoding: [0x40,0x80,A,0bAAAAAA00]
# CHECK-LE: bge 0, target                   # encoding: [0bAAAAAA00,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl target
# CHECK-BE: bgea 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA10]
# CHECK-LE: bgea 2, target                  # encoding: [0bAAAAAA10,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla 2, target
# CHECK-BE: bgea 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA10]
# CHECK-LE: bgea 0, target                  # encoding: [0bAAAAAA10,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla target
# CHECK-BE: bgelr 2                         # encoding: [0x4c,0x88,0x00,0x20]
# CHECK-LE: bgelr 2                         # encoding: [0x20,0x00,0x88,0x4c]
            bnllr 2
# CHECK-BE: bgelr 0                         # encoding: [0x4c,0x80,0x00,0x20]
# CHECK-LE: bgelr 0                         # encoding: [0x20,0x00,0x80,0x4c]
            bnllr
# CHECK-BE: bgectr 2                        # encoding: [0x4c,0x88,0x04,0x20]
# CHECK-LE: bgectr 2                        # encoding: [0x20,0x04,0x88,0x4c]
            bnlctr 2
# CHECK-BE: bgectr 0                        # encoding: [0x4c,0x80,0x04,0x20]
# CHECK-LE: bgectr 0                        # encoding: [0x20,0x04,0x80,0x4c]
            bnlctr
# CHECK-BE: bgel 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA01]
# CHECK-LE: bgel 2, target                  # encoding: [0bAAAAAA01,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll 2, target
# CHECK-BE: bgel 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA01]
# CHECK-LE: bgel 0, target                  # encoding: [0bAAAAAA01,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll target
# CHECK-BE: bgela 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA11]
# CHECK-LE: bgela 2, target                  # encoding: [0bAAAAAA11,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla 2, target
# CHECK-BE: bgela 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA11]
# CHECK-LE: bgela 0, target                  # encoding: [0bAAAAAA11,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla target
# CHECK-BE: bgelrl 2                        # encoding: [0x4c,0x88,0x00,0x21]
# CHECK-LE: bgelrl 2                        # encoding: [0x21,0x00,0x88,0x4c]
            bnllrl 2
# CHECK-BE: bgelrl 0                        # encoding: [0x4c,0x80,0x00,0x21]
# CHECK-LE: bgelrl 0                        # encoding: [0x21,0x00,0x80,0x4c]
            bnllrl
# CHECK-BE: bgectrl 2                       # encoding: [0x4c,0x88,0x04,0x21]
# CHECK-LE: bgectrl 2                       # encoding: [0x21,0x04,0x88,0x4c]
            bnlctrl 2
# CHECK-BE: bgectrl 0                       # encoding: [0x4c,0x80,0x04,0x21]
# CHECK-LE: bgectrl 0                       # encoding: [0x21,0x04,0x80,0x4c]
            bnlctrl

# CHECK-BE: bge+ 2, target                  # encoding: [0x40,0xe8,A,0bAAAAAA00]
# CHECK-LE: bge+ 2, target                  # encoding: [0bAAAAAA00,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl+ 2, target
# CHECK-BE: bge+ 0, target                  # encoding: [0x40,0xe0,A,0bAAAAAA00]
# CHECK-LE: bge+ 0, target                  # encoding: [0bAAAAAA00,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl+ target
# CHECK-BE: bgea+ 2, target                 # encoding: [0x40,0xe8,A,0bAAAAAA10]
# CHECK-LE: bgea+ 2, target                 # encoding: [0bAAAAAA10,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla+ 2, target
# CHECK-BE: bgea+ 0, target                 # encoding: [0x40,0xe0,A,0bAAAAAA10]
# CHECK-LE: bgea+ 0, target                 # encoding: [0bAAAAAA10,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla+ target
# CHECK-BE: bgelr+ 2                        # encoding: [0x4c,0xe8,0x00,0x20]
# CHECK-LE: bgelr+ 2                        # encoding: [0x20,0x00,0xe8,0x4c]
            bnllr+ 2
# CHECK-BE: bgelr+ 0                        # encoding: [0x4c,0xe0,0x00,0x20]
# CHECK-LE: bgelr+ 0                        # encoding: [0x20,0x00,0xe0,0x4c]
            bnllr+
# CHECK-BE: bgectr+ 2                       # encoding: [0x4c,0xe8,0x04,0x20]
# CHECK-LE: bgectr+ 2                       # encoding: [0x20,0x04,0xe8,0x4c]
            bnlctr+ 2
# CHECK-BE: bgectr+ 0                       # encoding: [0x4c,0xe0,0x04,0x20]
# CHECK-LE: bgectr+ 0                       # encoding: [0x20,0x04,0xe0,0x4c]
            bnlctr+
# CHECK-BE: bgel+ 2, target                 # encoding: [0x40,0xe8,A,0bAAAAAA01]
# CHECK-LE: bgel+ 2, target                 # encoding: [0bAAAAAA01,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll+ 2, target
# CHECK-BE: bgel+ 0, target                 # encoding: [0x40,0xe0,A,0bAAAAAA01]
# CHECK-LE: bgel+ 0, target                 # encoding: [0bAAAAAA01,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll+ target
# CHECK-BE: bgela+ 2, target                # encoding: [0x40,0xe8,A,0bAAAAAA11]
# CHECK-LE: bgela+ 2, target                # encoding: [0bAAAAAA11,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla+ 2, target
# CHECK-BE: bgela+ 0, target                # encoding: [0x40,0xe0,A,0bAAAAAA11]
# CHECK-LE: bgela+ 0, target                # encoding: [0bAAAAAA11,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla+ target
# CHECK-BE: bgelrl+ 2                       # encoding: [0x4c,0xe8,0x00,0x21]
# CHECK-LE: bgelrl+ 2                       # encoding: [0x21,0x00,0xe8,0x4c]
            bnllrl+ 2
# CHECK-BE: bgelrl+ 0                       # encoding: [0x4c,0xe0,0x00,0x21]
# CHECK-LE: bgelrl+ 0                       # encoding: [0x21,0x00,0xe0,0x4c]
            bnllrl+
# CHECK-BE: bgectrl+ 2                      # encoding: [0x4c,0xe8,0x04,0x21]
# CHECK-LE: bgectrl+ 2                      # encoding: [0x21,0x04,0xe8,0x4c]
            bnlctrl+ 2
# CHECK-BE: bgectrl+ 0                      # encoding: [0x4c,0xe0,0x04,0x21]
# CHECK-LE: bgectrl+ 0                      # encoding: [0x21,0x04,0xe0,0x4c]
            bnlctrl+

# CHECK-BE: bge- 2, target                  # encoding: [0x40,0xc8,A,0bAAAAAA00]
# CHECK-LE: bge- 2, target                  # encoding: [0bAAAAAA00,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl- 2, target
# CHECK-BE: bge- 0, target                  # encoding: [0x40,0xc0,A,0bAAAAAA00]
# CHECK-LE: bge- 0, target                  # encoding: [0bAAAAAA00,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl- target
# CHECK-BE: bgea- 2, target                 # encoding: [0x40,0xc8,A,0bAAAAAA10]
# CHECK-LE: bgea- 2, target                 # encoding: [0bAAAAAA10,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla- 2, target
# CHECK-BE: bgea- 0, target                 # encoding: [0x40,0xc0,A,0bAAAAAA10]
# CHECK-LE: bgea- 0, target                 # encoding: [0bAAAAAA10,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla- target
# CHECK-BE: bgelr- 2                        # encoding: [0x4c,0xc8,0x00,0x20]
# CHECK-LE: bgelr- 2                        # encoding: [0x20,0x00,0xc8,0x4c]
            bnllr- 2
# CHECK-BE: bgelr- 0                        # encoding: [0x4c,0xc0,0x00,0x20]
# CHECK-LE: bgelr- 0                        # encoding: [0x20,0x00,0xc0,0x4c]
            bnllr-
# CHECK-BE: bgectr- 2                       # encoding: [0x4c,0xc8,0x04,0x20]
# CHECK-LE: bgectr- 2                       # encoding: [0x20,0x04,0xc8,0x4c]
            bnlctr- 2
# CHECK-BE: bgectr- 0                       # encoding: [0x4c,0xc0,0x04,0x20]
# CHECK-LE: bgectr- 0                       # encoding: [0x20,0x04,0xc0,0x4c]
            bnlctr-
# CHECK-BE: bgel- 2, target                 # encoding: [0x40,0xc8,A,0bAAAAAA01]
# CHECK-LE: bgel- 2, target                 # encoding: [0bAAAAAA01,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll- 2, target
# CHECK-BE: bgel- 0, target                 # encoding: [0x40,0xc0,A,0bAAAAAA01]
# CHECK-LE: bgel- 0, target                 # encoding: [0bAAAAAA01,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll- target
# CHECK-BE: bgela- 2, target                # encoding: [0x40,0xc8,A,0bAAAAAA11]
# CHECK-LE: bgela- 2, target                # encoding: [0bAAAAAA11,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla- 2, target
# CHECK-BE: bgela- 0, target                # encoding: [0x40,0xc0,A,0bAAAAAA11]
# CHECK-LE: bgela- 0, target                # encoding: [0bAAAAAA11,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla- target
# CHECK-BE: bgelrl- 2                       # encoding: [0x4c,0xc8,0x00,0x21]
# CHECK-LE: bgelrl- 2                       # encoding: [0x21,0x00,0xc8,0x4c]
            bnllrl- 2
# CHECK-BE: bgelrl- 0                       # encoding: [0x4c,0xc0,0x00,0x21]
# CHECK-LE: bgelrl- 0                       # encoding: [0x21,0x00,0xc0,0x4c]
            bnllrl-
# CHECK-BE: bgectrl- 2                      # encoding: [0x4c,0xc8,0x04,0x21]
# CHECK-LE: bgectrl- 2                      # encoding: [0x21,0x04,0xc8,0x4c]
            bnlctrl- 2
# CHECK-BE: bgectrl- 0                      # encoding: [0x4c,0xc0,0x04,0x21]
# CHECK-LE: bgectrl- 0                      # encoding: [0x21,0x04,0xc0,0x4c]
            bnlctrl-

# CHECK-BE: bne 2, target                   # encoding: [0x40,0x8a,A,0bAAAAAA00]
# CHECK-LE: bne 2, target                   # encoding: [0bAAAAAA00,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne 2, target
# CHECK-BE: bne 0, target                   # encoding: [0x40,0x82,A,0bAAAAAA00]
# CHECK-LE: bne 0, target                   # encoding: [0bAAAAAA00,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne target
# CHECK-BE: bnea 2, target                  # encoding: [0x40,0x8a,A,0bAAAAAA10]
# CHECK-LE: bnea 2, target                  # encoding: [0bAAAAAA10,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea 2, target
# CHECK-BE: bnea 0, target                  # encoding: [0x40,0x82,A,0bAAAAAA10]
# CHECK-LE: bnea 0, target                  # encoding: [0bAAAAAA10,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea target
# CHECK-BE: bnelr 2                         # encoding: [0x4c,0x8a,0x00,0x20]
# CHECK-LE: bnelr 2                         # encoding: [0x20,0x00,0x8a,0x4c]
            bnelr 2
# CHECK-BE: bnelr 0                         # encoding: [0x4c,0x82,0x00,0x20]
# CHECK-LE: bnelr 0                         # encoding: [0x20,0x00,0x82,0x4c]
            bnelr
# CHECK-BE: bnectr 2                        # encoding: [0x4c,0x8a,0x04,0x20]
# CHECK-LE: bnectr 2                        # encoding: [0x20,0x04,0x8a,0x4c]
            bnectr 2
# CHECK-BE: bnectr 0                        # encoding: [0x4c,0x82,0x04,0x20]
# CHECK-LE: bnectr 0                        # encoding: [0x20,0x04,0x82,0x4c]
            bnectr
# CHECK-BE: bnel 2, target                  # encoding: [0x40,0x8a,A,0bAAAAAA01]
# CHECK-LE: bnel 2, target                  # encoding: [0bAAAAAA01,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel 2, target
# CHECK-BE: bnel 0, target                  # encoding: [0x40,0x82,A,0bAAAAAA01]
# CHECK-LE: bnel 0, target                  # encoding: [0bAAAAAA01,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel target
# CHECK-BE: bnela 2, target                 # encoding: [0x40,0x8a,A,0bAAAAAA11]
# CHECK-LE: bnela 2, target                 # encoding: [0bAAAAAA11,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela 2, target
# CHECK-BE: bnela 0, target                 # encoding: [0x40,0x82,A,0bAAAAAA11]
# CHECK-LE: bnela 0, target                 # encoding: [0bAAAAAA11,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela target
# CHECK-BE: bnelrl 2                        # encoding: [0x4c,0x8a,0x00,0x21]
# CHECK-LE: bnelrl 2                        # encoding: [0x21,0x00,0x8a,0x4c]
            bnelrl 2
# CHECK-BE: bnelrl 0                        # encoding: [0x4c,0x82,0x00,0x21]
# CHECK-LE: bnelrl 0                        # encoding: [0x21,0x00,0x82,0x4c]
            bnelrl
# CHECK-BE: bnectrl 2                       # encoding: [0x4c,0x8a,0x04,0x21]
# CHECK-LE: bnectrl 2                       # encoding: [0x21,0x04,0x8a,0x4c]
            bnectrl 2
# CHECK-BE: bnectrl 0                       # encoding: [0x4c,0x82,0x04,0x21]
# CHECK-LE: bnectrl 0                       # encoding: [0x21,0x04,0x82,0x4c]
            bnectrl

# CHECK-BE: bne+ 2, target                  # encoding: [0x40,0xea,A,0bAAAAAA00]
# CHECK-LE: bne+ 2, target                  # encoding: [0bAAAAAA00,A,0xea,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne+ 2, target
# CHECK-BE: bne+ 0, target                  # encoding: [0x40,0xe2,A,0bAAAAAA00]
# CHECK-LE: bne+ 0, target                  # encoding: [0bAAAAAA00,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne+ target
# CHECK-BE: bnea+ 2, target                 # encoding: [0x40,0xea,A,0bAAAAAA10]
# CHECK-LE: bnea+ 2, target                 # encoding: [0bAAAAAA10,A,0xea,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea+ 2, target
# CHECK-BE: bnea+ 0, target                 # encoding: [0x40,0xe2,A,0bAAAAAA10]
# CHECK-LE: bnea+ 0, target                 # encoding: [0bAAAAAA10,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea+ target
# CHECK-BE: bnelr+ 2                        # encoding: [0x4c,0xea,0x00,0x20]
# CHECK-LE: bnelr+ 2                        # encoding: [0x20,0x00,0xea,0x4c]
            bnelr+ 2
# CHECK-BE: bnelr+ 0                        # encoding: [0x4c,0xe2,0x00,0x20]
# CHECK-LE: bnelr+ 0                        # encoding: [0x20,0x00,0xe2,0x4c]
            bnelr+
# CHECK-BE: bnectr+ 2                       # encoding: [0x4c,0xea,0x04,0x20]
# CHECK-LE: bnectr+ 2                       # encoding: [0x20,0x04,0xea,0x4c]
            bnectr+ 2
# CHECK-BE: bnectr+ 0                       # encoding: [0x4c,0xe2,0x04,0x20]
# CHECK-LE: bnectr+ 0                       # encoding: [0x20,0x04,0xe2,0x4c]
            bnectr+
# CHECK-BE: bnel+ 2, target                 # encoding: [0x40,0xea,A,0bAAAAAA01]
# CHECK-LE: bnel+ 2, target                 # encoding: [0bAAAAAA01,A,0xea,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel+ 2, target
# CHECK-BE: bnel+ 0, target                 # encoding: [0x40,0xe2,A,0bAAAAAA01]
# CHECK-LE: bnel+ 0, target                 # encoding: [0bAAAAAA01,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel+ target
# CHECK-BE: bnela+ 2, target                # encoding: [0x40,0xea,A,0bAAAAAA11]
# CHECK-LE: bnela+ 2, target                # encoding: [0bAAAAAA11,A,0xea,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela+ 2, target
# CHECK-BE: bnela+ 0, target                # encoding: [0x40,0xe2,A,0bAAAAAA11]
# CHECK-LE: bnela+ 0, target                # encoding: [0bAAAAAA11,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela+ target
# CHECK-BE: bnelrl+ 2                       # encoding: [0x4c,0xea,0x00,0x21]
# CHECK-LE: bnelrl+ 2                       # encoding: [0x21,0x00,0xea,0x4c]
            bnelrl+ 2
# CHECK-BE: bnelrl+ 0                       # encoding: [0x4c,0xe2,0x00,0x21]
# CHECK-LE: bnelrl+ 0                       # encoding: [0x21,0x00,0xe2,0x4c]
            bnelrl+
# CHECK-BE: bnectrl+ 2                      # encoding: [0x4c,0xea,0x04,0x21]
# CHECK-LE: bnectrl+ 2                      # encoding: [0x21,0x04,0xea,0x4c]
            bnectrl+ 2
# CHECK-BE: bnectrl+ 0                      # encoding: [0x4c,0xe2,0x04,0x21]
# CHECK-LE: bnectrl+ 0                      # encoding: [0x21,0x04,0xe2,0x4c]
            bnectrl+

# CHECK-BE: bne- 2, target                  # encoding: [0x40,0xca,A,0bAAAAAA00]
# CHECK-LE: bne- 2, target                  # encoding: [0bAAAAAA00,A,0xca,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne- 2, target
# CHECK-BE: bne- 0, target                  # encoding: [0x40,0xc2,A,0bAAAAAA00]
# CHECK-LE: bne- 0, target                  # encoding: [0bAAAAAA00,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne- target
# CHECK-BE: bnea- 2, target                 # encoding: [0x40,0xca,A,0bAAAAAA10]
# CHECK-LE: bnea- 2, target                 # encoding: [0bAAAAAA10,A,0xca,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea- 2, target
# CHECK-BE: bnea- 0, target                 # encoding: [0x40,0xc2,A,0bAAAAAA10]
# CHECK-LE: bnea- 0, target                 # encoding: [0bAAAAAA10,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea- target
# CHECK-BE: bnelr- 2                        # encoding: [0x4c,0xca,0x00,0x20]
# CHECK-LE: bnelr- 2                        # encoding: [0x20,0x00,0xca,0x4c]
            bnelr- 2
# CHECK-BE: bnelr- 0                        # encoding: [0x4c,0xc2,0x00,0x20]
# CHECK-LE: bnelr- 0                        # encoding: [0x20,0x00,0xc2,0x4c]
            bnelr-
# CHECK-BE: bnectr- 2                       # encoding: [0x4c,0xca,0x04,0x20]
# CHECK-LE: bnectr- 2                       # encoding: [0x20,0x04,0xca,0x4c]
            bnectr- 2
# CHECK-BE: bnectr- 0                       # encoding: [0x4c,0xc2,0x04,0x20]
# CHECK-LE: bnectr- 0                       # encoding: [0x20,0x04,0xc2,0x4c]
            bnectr-
# CHECK-BE: bnel- 2, target                 # encoding: [0x40,0xca,A,0bAAAAAA01]
# CHECK-LE: bnel- 2, target                 # encoding: [0bAAAAAA01,A,0xca,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel- 2, target
# CHECK-BE: bnel- 0, target                 # encoding: [0x40,0xc2,A,0bAAAAAA01]
# CHECK-LE: bnel- 0, target                 # encoding: [0bAAAAAA01,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel- target
# CHECK-BE: bnela- 2, target                # encoding: [0x40,0xca,A,0bAAAAAA11]
# CHECK-LE: bnela- 2, target                # encoding: [0bAAAAAA11,A,0xca,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela- 2, target
# CHECK-BE: bnela- 0, target                # encoding: [0x40,0xc2,A,0bAAAAAA11]
# CHECK-LE: bnela- 0, target                # encoding: [0bAAAAAA11,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela- target
# CHECK-BE: bnelrl- 2                       # encoding: [0x4c,0xca,0x00,0x21]
# CHECK-LE: bnelrl- 2                       # encoding: [0x21,0x00,0xca,0x4c]
            bnelrl- 2
# CHECK-BE: bnelrl- 0                       # encoding: [0x4c,0xc2,0x00,0x21]
# CHECK-LE: bnelrl- 0                       # encoding: [0x21,0x00,0xc2,0x4c]
            bnelrl-
# CHECK-BE: bnectrl- 2                      # encoding: [0x4c,0xca,0x04,0x21]
# CHECK-LE: bnectrl- 2                      # encoding: [0x21,0x04,0xca,0x4c]
            bnectrl- 2
# CHECK-BE: bnectrl- 0                      # encoding: [0x4c,0xc2,0x04,0x21]
# CHECK-LE: bnectrl- 0                      # encoding: [0x21,0x04,0xc2,0x4c]
            bnectrl-

# CHECK-BE: ble 2, target                   # encoding: [0x40,0x89,A,0bAAAAAA00]
# CHECK-LE: ble 2, target                   # encoding: [0bAAAAAA00,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng 2, target
# CHECK-BE: ble 0, target                   # encoding: [0x40,0x81,A,0bAAAAAA00]
# CHECK-LE: ble 0, target                   # encoding: [0bAAAAAA00,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng target
# CHECK-BE: blea 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA10]
# CHECK-LE: blea 2, target                  # encoding: [0bAAAAAA10,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga 2, target
# CHECK-BE: blea 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA10]
# CHECK-LE: blea 0, target                  # encoding: [0bAAAAAA10,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga target
# CHECK-BE: blelr 2                         # encoding: [0x4c,0x89,0x00,0x20]
# CHECK-LE: blelr 2                         # encoding: [0x20,0x00,0x89,0x4c]
            bnglr 2
# CHECK-BE: blelr 0                         # encoding: [0x4c,0x81,0x00,0x20]
# CHECK-LE: blelr 0                         # encoding: [0x20,0x00,0x81,0x4c]
            bnglr
# CHECK-BE: blectr 2                        # encoding: [0x4c,0x89,0x04,0x20]
# CHECK-LE: blectr 2                        # encoding: [0x20,0x04,0x89,0x4c]
            bngctr 2
# CHECK-BE: blectr 0                        # encoding: [0x4c,0x81,0x04,0x20]
# CHECK-LE: blectr 0                        # encoding: [0x20,0x04,0x81,0x4c]
            bngctr
# CHECK-BE: blel 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA01]
# CHECK-LE: blel 2, target                  # encoding: [0bAAAAAA01,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl 2, target
# CHECK-BE: blel 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA01]
# CHECK-LE: blel 0, target                  # encoding: [0bAAAAAA01,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl target
# CHECK-BE: blela 2, target                 # encoding: [0x40,0x89,A,0bAAAAAA11]
# CHECK-LE: blela 2, target                 # encoding: [0bAAAAAA11,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla 2, target
# CHECK-BE: blela 0, target                 # encoding: [0x40,0x81,A,0bAAAAAA11]
# CHECK-LE: blela 0, target                 # encoding: [0bAAAAAA11,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla target
# CHECK-BE: blelrl 2                        # encoding: [0x4c,0x89,0x00,0x21]
# CHECK-LE: blelrl 2                        # encoding: [0x21,0x00,0x89,0x4c]
            bnglrl 2
# CHECK-BE: blelrl 0                        # encoding: [0x4c,0x81,0x00,0x21]
# CHECK-LE: blelrl 0                        # encoding: [0x21,0x00,0x81,0x4c]
            bnglrl
# CHECK-BE: blectrl 2                       # encoding: [0x4c,0x89,0x04,0x21]
# CHECK-LE: blectrl 2                       # encoding: [0x21,0x04,0x89,0x4c]
            bngctrl 2
# CHECK-BE: blectrl 0                       # encoding: [0x4c,0x81,0x04,0x21]
# CHECK-LE: blectrl 0                       # encoding: [0x21,0x04,0x81,0x4c]
            bngctrl

# CHECK-BE: ble+ 2, target                   # encoding: [0x40,0xe9,A,0bAAAAAA00]
# CHECK-LE: ble+ 2, target                   # encoding: [0bAAAAAA00,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng+ 2, target
# CHECK-BE: ble+ 0, target                   # encoding: [0x40,0xe1,A,0bAAAAAA00]
# CHECK-LE: ble+ 0, target                   # encoding: [0bAAAAAA00,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng+ target
# CHECK-BE: blea+ 2, target                  # encoding: [0x40,0xe9,A,0bAAAAAA10]
# CHECK-LE: blea+ 2, target                  # encoding: [0bAAAAAA10,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga+ 2, target
# CHECK-BE: blea+ 0, target                  # encoding: [0x40,0xe1,A,0bAAAAAA10]
# CHECK-LE: blea+ 0, target                  # encoding: [0bAAAAAA10,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga+ target
# CHECK-BE: blelr+ 2                         # encoding: [0x4c,0xe9,0x00,0x20]
# CHECK-LE: blelr+ 2                         # encoding: [0x20,0x00,0xe9,0x4c]
            bnglr+ 2
# CHECK-BE: blelr+ 0                         # encoding: [0x4c,0xe1,0x00,0x20]
# CHECK-LE: blelr+ 0                         # encoding: [0x20,0x00,0xe1,0x4c]
            bnglr+
# CHECK-BE: blectr+ 2                        # encoding: [0x4c,0xe9,0x04,0x20]
# CHECK-LE: blectr+ 2                        # encoding: [0x20,0x04,0xe9,0x4c]
            bngctr+ 2
# CHECK-BE: blectr+ 0                        # encoding: [0x4c,0xe1,0x04,0x20]
# CHECK-LE: blectr+ 0                        # encoding: [0x20,0x04,0xe1,0x4c]
            bngctr+
# CHECK-BE: blel+ 2, target                  # encoding: [0x40,0xe9,A,0bAAAAAA01]
# CHECK-LE: blel+ 2, target                  # encoding: [0bAAAAAA01,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl+ 2, target
# CHECK-BE: blel+ 0, target                  # encoding: [0x40,0xe1,A,0bAAAAAA01]
# CHECK-LE: blel+ 0, target                  # encoding: [0bAAAAAA01,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl+ target
# CHECK-BE: blela+ 2, target                 # encoding: [0x40,0xe9,A,0bAAAAAA11]
# CHECK-LE: blela+ 2, target                 # encoding: [0bAAAAAA11,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla+ 2, target
# CHECK-BE: blela+ 0, target                 # encoding: [0x40,0xe1,A,0bAAAAAA11]
# CHECK-LE: blela+ 0, target                 # encoding: [0bAAAAAA11,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla+ target
# CHECK-BE: blelrl+ 2                        # encoding: [0x4c,0xe9,0x00,0x21]
# CHECK-LE: blelrl+ 2                        # encoding: [0x21,0x00,0xe9,0x4c]
            bnglrl+ 2
# CHECK-BE: blelrl+ 0                        # encoding: [0x4c,0xe1,0x00,0x21]
# CHECK-LE: blelrl+ 0                        # encoding: [0x21,0x00,0xe1,0x4c]
            bnglrl+
# CHECK-BE: blectrl+ 2                       # encoding: [0x4c,0xe9,0x04,0x21]
# CHECK-LE: blectrl+ 2                       # encoding: [0x21,0x04,0xe9,0x4c]
            bngctrl+ 2
# CHECK-BE: blectrl+ 0                       # encoding: [0x4c,0xe1,0x04,0x21]
# CHECK-LE: blectrl+ 0                       # encoding: [0x21,0x04,0xe1,0x4c]
            bngctrl+

# CHECK-BE: ble- 2, target                   # encoding: [0x40,0xc9,A,0bAAAAAA00]
# CHECK-LE: ble- 2, target                   # encoding: [0bAAAAAA00,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng- 2, target
# CHECK-BE: ble- 0, target                   # encoding: [0x40,0xc1,A,0bAAAAAA00]
# CHECK-LE: ble- 0, target                   # encoding: [0bAAAAAA00,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng- target
# CHECK-BE: blea- 2, target                  # encoding: [0x40,0xc9,A,0bAAAAAA10]
# CHECK-LE: blea- 2, target                  # encoding: [0bAAAAAA10,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga- 2, target
# CHECK-BE: blea- 0, target                  # encoding: [0x40,0xc1,A,0bAAAAAA10]
# CHECK-LE: blea- 0, target                  # encoding: [0bAAAAAA10,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga- target
# CHECK-BE: blelr- 2                         # encoding: [0x4c,0xc9,0x00,0x20]
# CHECK-LE: blelr- 2                         # encoding: [0x20,0x00,0xc9,0x4c]
            bnglr- 2
# CHECK-BE: blelr- 0                         # encoding: [0x4c,0xc1,0x00,0x20]
# CHECK-LE: blelr- 0                         # encoding: [0x20,0x00,0xc1,0x4c]
            bnglr-
# CHECK-BE: blectr- 2                        # encoding: [0x4c,0xc9,0x04,0x20]
# CHECK-LE: blectr- 2                        # encoding: [0x20,0x04,0xc9,0x4c]
            bngctr- 2
# CHECK-BE: blectr- 0                        # encoding: [0x4c,0xc1,0x04,0x20]
# CHECK-LE: blectr- 0                        # encoding: [0x20,0x04,0xc1,0x4c]
            bngctr-
# CHECK-BE: blel- 2, target                  # encoding: [0x40,0xc9,A,0bAAAAAA01]
# CHECK-LE: blel- 2, target                  # encoding: [0bAAAAAA01,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl- 2, target
# CHECK-BE: blel- 0, target                  # encoding: [0x40,0xc1,A,0bAAAAAA01]
# CHECK-LE: blel- 0, target                  # encoding: [0bAAAAAA01,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl- target
# CHECK-BE: blela- 2, target                 # encoding: [0x40,0xc9,A,0bAAAAAA11]
# CHECK-LE: blela- 2, target                 # encoding: [0bAAAAAA11,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla- 2, target
# CHECK-BE: blela- 0, target                 # encoding: [0x40,0xc1,A,0bAAAAAA11]
# CHECK-LE: blela- 0, target                 # encoding: [0bAAAAAA11,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla- target
# CHECK-BE: blelrl- 2                        # encoding: [0x4c,0xc9,0x00,0x21]
# CHECK-LE: blelrl- 2                        # encoding: [0x21,0x00,0xc9,0x4c]
            bnglrl- 2
# CHECK-BE: blelrl- 0                        # encoding: [0x4c,0xc1,0x00,0x21]
# CHECK-LE: blelrl- 0                        # encoding: [0x21,0x00,0xc1,0x4c]
            bnglrl-
# CHECK-BE: blectrl- 2                       # encoding: [0x4c,0xc9,0x04,0x21]
# CHECK-LE: blectrl- 2                       # encoding: [0x21,0x04,0xc9,0x4c]
            bngctrl- 2
# CHECK-BE: blectrl- 0                       # encoding: [0x4c,0xc1,0x04,0x21]
# CHECK-LE: blectrl- 0                       # encoding: [0x21,0x04,0xc1,0x4c]
            bngctrl-

# CHECK-BE: bun 2, target                   # encoding: [0x41,0x8b,A,0bAAAAAA00]
# CHECK-LE: bun 2, target                   # encoding: [0bAAAAAA00,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso 2, target
# CHECK-BE: bun 0, target                   # encoding: [0x41,0x83,A,0bAAAAAA00]
# CHECK-LE: bun 0, target                   # encoding: [0bAAAAAA00,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso target
# CHECK-BE: buna 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA10]
# CHECK-LE: buna 2, target                  # encoding: [0bAAAAAA10,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa 2, target
# CHECK-BE: buna 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA10]
# CHECK-LE: buna 0, target                  # encoding: [0bAAAAAA10,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa target
# CHECK-BE: bunlr 2                         # encoding: [0x4d,0x8b,0x00,0x20]
# CHECK-LE: bunlr 2                         # encoding: [0x20,0x00,0x8b,0x4d]
            bsolr 2
# CHECK-BE: bunlr 0                         # encoding: [0x4d,0x83,0x00,0x20]
# CHECK-LE: bunlr 0                         # encoding: [0x20,0x00,0x83,0x4d]
            bsolr
# CHECK-BE: bunctr 2                        # encoding: [0x4d,0x8b,0x04,0x20]
# CHECK-LE: bunctr 2                        # encoding: [0x20,0x04,0x8b,0x4d]
            bsoctr 2
# CHECK-BE: bunctr 0                        # encoding: [0x4d,0x83,0x04,0x20]
# CHECK-LE: bunctr 0                        # encoding: [0x20,0x04,0x83,0x4d]
            bsoctr
# CHECK-BE: bunl 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA01]
# CHECK-LE: bunl 2, target                  # encoding: [0bAAAAAA01,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol 2, target
# CHECK-BE: bunl 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA01]
# CHECK-LE: bunl 0, target                  # encoding: [0bAAAAAA01,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol target
# CHECK-BE: bunla 2, target                 # encoding: [0x41,0x8b,A,0bAAAAAA11]
# CHECK-LE: bunla 2, target                 # encoding: [0bAAAAAA11,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola 2, target
# CHECK-BE: bunla 0, target                 # encoding: [0x41,0x83,A,0bAAAAAA11]
# CHECK-LE: bunla 0, target                 # encoding: [0bAAAAAA11,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola target
# CHECK-BE: bunlrl 2                        # encoding: [0x4d,0x8b,0x00,0x21]
# CHECK-LE: bunlrl 2                        # encoding: [0x21,0x00,0x8b,0x4d]
            bsolrl 2
# CHECK-BE: bunlrl 0                        # encoding: [0x4d,0x83,0x00,0x21]
# CHECK-LE: bunlrl 0                        # encoding: [0x21,0x00,0x83,0x4d]
            bsolrl
# CHECK-BE: bunctrl 2                       # encoding: [0x4d,0x8b,0x04,0x21]
# CHECK-LE: bunctrl 2                       # encoding: [0x21,0x04,0x8b,0x4d]
            bsoctrl 2
# CHECK-BE: bunctrl 0                       # encoding: [0x4d,0x83,0x04,0x21]
# CHECK-LE: bunctrl 0                       # encoding: [0x21,0x04,0x83,0x4d]
            bsoctrl

# CHECK-BE: bun+ 2, target                  # encoding: [0x41,0xeb,A,0bAAAAAA00]
# CHECK-LE: bun+ 2, target                  # encoding: [0bAAAAAA00,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso+ 2, target
# CHECK-BE: bun+ 0, target                  # encoding: [0x41,0xe3,A,0bAAAAAA00]
# CHECK-LE: bun+ 0, target                  # encoding: [0bAAAAAA00,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso+ target
# CHECK-BE: buna+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA10]
# CHECK-LE: buna+ 2, target                 # encoding: [0bAAAAAA10,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa+ 2, target
# CHECK-BE: buna+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA10]
# CHECK-LE: buna+ 0, target                 # encoding: [0bAAAAAA10,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa+ target
# CHECK-BE: bunlr+ 2                        # encoding: [0x4d,0xeb,0x00,0x20]
# CHECK-LE: bunlr+ 2                        # encoding: [0x20,0x00,0xeb,0x4d]
            bsolr+ 2
# CHECK-BE: bunlr+ 0                        # encoding: [0x4d,0xe3,0x00,0x20]
# CHECK-LE: bunlr+ 0                        # encoding: [0x20,0x00,0xe3,0x4d]
            bsolr+
# CHECK-BE: bunctr+ 2                       # encoding: [0x4d,0xeb,0x04,0x20]
# CHECK-LE: bunctr+ 2                       # encoding: [0x20,0x04,0xeb,0x4d]
            bsoctr+ 2
# CHECK-BE: bunctr+ 0                       # encoding: [0x4d,0xe3,0x04,0x20]
# CHECK-LE: bunctr+ 0                       # encoding: [0x20,0x04,0xe3,0x4d]
            bsoctr+
# CHECK-BE: bunl+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA01]
# CHECK-LE: bunl+ 2, target                 # encoding: [0bAAAAAA01,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol+ 2, target
# CHECK-BE: bunl+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA01]
# CHECK-LE: bunl+ 0, target                 # encoding: [0bAAAAAA01,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol+ target
# CHECK-BE: bunla+ 2, target                # encoding: [0x41,0xeb,A,0bAAAAAA11]
# CHECK-LE: bunla+ 2, target                # encoding: [0bAAAAAA11,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola+ 2, target
# CHECK-BE: bunla+ 0, target                # encoding: [0x41,0xe3,A,0bAAAAAA11]
# CHECK-LE: bunla+ 0, target                # encoding: [0bAAAAAA11,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola+ target
# CHECK-BE: bunlrl+ 2                       # encoding: [0x4d,0xeb,0x00,0x21]
# CHECK-LE: bunlrl+ 2                       # encoding: [0x21,0x00,0xeb,0x4d]
            bsolrl+ 2
# CHECK-BE: bunlrl+ 0                       # encoding: [0x4d,0xe3,0x00,0x21]
# CHECK-LE: bunlrl+ 0                       # encoding: [0x21,0x00,0xe3,0x4d]
            bsolrl+
# CHECK-BE: bunctrl+ 2                      # encoding: [0x4d,0xeb,0x04,0x21]
# CHECK-LE: bunctrl+ 2                      # encoding: [0x21,0x04,0xeb,0x4d]
            bsoctrl+ 2
# CHECK-BE: bunctrl+ 0                      # encoding: [0x4d,0xe3,0x04,0x21]
# CHECK-LE: bunctrl+ 0                      # encoding: [0x21,0x04,0xe3,0x4d]
            bsoctrl+

# CHECK-BE: bun- 2, target                  # encoding: [0x41,0xcb,A,0bAAAAAA00]
# CHECK-LE: bun- 2, target                  # encoding: [0bAAAAAA00,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso- 2, target
# CHECK-BE: bun- 0, target                  # encoding: [0x41,0xc3,A,0bAAAAAA00]
# CHECK-LE: bun- 0, target                  # encoding: [0bAAAAAA00,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso- target
# CHECK-BE: buna- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA10]
# CHECK-LE: buna- 2, target                 # encoding: [0bAAAAAA10,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa- 2, target
# CHECK-BE: buna- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA10]
# CHECK-LE: buna- 0, target                 # encoding: [0bAAAAAA10,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa- target
# CHECK-BE: bunlr- 2                        # encoding: [0x4d,0xcb,0x00,0x20]
# CHECK-LE: bunlr- 2                        # encoding: [0x20,0x00,0xcb,0x4d]
            bsolr- 2
# CHECK-BE: bunlr- 0                        # encoding: [0x4d,0xc3,0x00,0x20]
# CHECK-LE: bunlr- 0                        # encoding: [0x20,0x00,0xc3,0x4d]
            bsolr-
# CHECK-BE: bunctr- 2                       # encoding: [0x4d,0xcb,0x04,0x20]
# CHECK-LE: bunctr- 2                       # encoding: [0x20,0x04,0xcb,0x4d]
            bsoctr- 2
# CHECK-BE: bunctr- 0                       # encoding: [0x4d,0xc3,0x04,0x20]
# CHECK-LE: bunctr- 0                       # encoding: [0x20,0x04,0xc3,0x4d]
            bsoctr-
# CHECK-BE: bunl- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA01]
# CHECK-LE: bunl- 2, target                 # encoding: [0bAAAAAA01,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol- 2, target
# CHECK-BE: bunl- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA01]
# CHECK-LE: bunl- 0, target                 # encoding: [0bAAAAAA01,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol- target
# CHECK-BE: bunla- 2, target                # encoding: [0x41,0xcb,A,0bAAAAAA11]
# CHECK-LE: bunla- 2, target                # encoding: [0bAAAAAA11,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola- 2, target
# CHECK-BE: bunla- 0, target                # encoding: [0x41,0xc3,A,0bAAAAAA11]
# CHECK-LE: bunla- 0, target                # encoding: [0bAAAAAA11,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola- target
# CHECK-BE: bunlrl- 2                       # encoding: [0x4d,0xcb,0x00,0x21]
# CHECK-LE: bunlrl- 2                       # encoding: [0x21,0x00,0xcb,0x4d]
            bsolrl- 2
# CHECK-BE: bunlrl- 0                       # encoding: [0x4d,0xc3,0x00,0x21]
# CHECK-LE: bunlrl- 0                       # encoding: [0x21,0x00,0xc3,0x4d]
            bsolrl-
# CHECK-BE: bunctrl- 2                      # encoding: [0x4d,0xcb,0x04,0x21]
# CHECK-LE: bunctrl- 2                      # encoding: [0x21,0x04,0xcb,0x4d]
            bsoctrl- 2
# CHECK-BE: bunctrl- 0                      # encoding: [0x4d,0xc3,0x04,0x21]
# CHECK-LE: bunctrl- 0                      # encoding: [0x21,0x04,0xc3,0x4d]
            bsoctrl-

# CHECK-BE: bnu 2, target                   # encoding: [0x40,0x8b,A,0bAAAAAA00]
# CHECK-LE: bnu 2, target                   # encoding: [0bAAAAAA00,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns 2, target
# CHECK-BE: bnu 0, target                   # encoding: [0x40,0x83,A,0bAAAAAA00]
# CHECK-LE: bnu 0, target                   # encoding: [0bAAAAAA00,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns target
# CHECK-BE: bnua 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA10]
# CHECK-LE: bnua 2, target                  # encoding: [0bAAAAAA10,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa 2, target
# CHECK-BE: bnua 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA10]
# CHECK-LE: bnua 0, target                  # encoding: [0bAAAAAA10,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa target
# CHECK-BE: bnulr 2                         # encoding: [0x4c,0x8b,0x00,0x20]
# CHECK-LE: bnulr 2                         # encoding: [0x20,0x00,0x8b,0x4c]
            bnslr 2
# CHECK-BE: bnulr 0                         # encoding: [0x4c,0x83,0x00,0x20]
# CHECK-LE: bnulr 0                         # encoding: [0x20,0x00,0x83,0x4c]
            bnslr
# CHECK-BE: bnuctr 2                        # encoding: [0x4c,0x8b,0x04,0x20]
# CHECK-LE: bnuctr 2                        # encoding: [0x20,0x04,0x8b,0x4c]
            bnsctr 2
# CHECK-BE: bnuctr 0                        # encoding: [0x4c,0x83,0x04,0x20]
# CHECK-LE: bnuctr 0                        # encoding: [0x20,0x04,0x83,0x4c]
            bnsctr
# CHECK-BE: bnul 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA01]
# CHECK-LE: bnul 2, target                  # encoding: [0bAAAAAA01,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl 2, target
# CHECK-BE: bnul 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA01]
# CHECK-LE: bnul 0, target                  # encoding: [0bAAAAAA01,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl target
# CHECK-BE: bnula 2, target                 # encoding: [0x40,0x8b,A,0bAAAAAA11]
# CHECK-LE: bnula 2, target                 # encoding: [0bAAAAAA11,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla 2, target
# CHECK-BE: bnula 0, target                 # encoding: [0x40,0x83,A,0bAAAAAA11]
# CHECK-LE: bnula 0, target                 # encoding: [0bAAAAAA11,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla target
# CHECK-BE: bnulrl 2                        # encoding: [0x4c,0x8b,0x00,0x21]
# CHECK-LE: bnulrl 2                        # encoding: [0x21,0x00,0x8b,0x4c]
            bnslrl 2
# CHECK-BE: bnulrl 0                        # encoding: [0x4c,0x83,0x00,0x21]
# CHECK-LE: bnulrl 0                        # encoding: [0x21,0x00,0x83,0x4c]
            bnslrl
# CHECK-BE: bnuctrl 2                       # encoding: [0x4c,0x8b,0x04,0x21]
# CHECK-LE: bnuctrl 2                       # encoding: [0x21,0x04,0x8b,0x4c]
            bnsctrl 2
# CHECK-BE: bnuctrl 0                       # encoding: [0x4c,0x83,0x04,0x21]
# CHECK-LE: bnuctrl 0                       # encoding: [0x21,0x04,0x83,0x4c]
            bnsctrl

# CHECK-BE: bnu+ 2, target                  # encoding: [0x40,0xeb,A,0bAAAAAA00]
# CHECK-LE: bnu+ 2, target                  # encoding: [0bAAAAAA00,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns+ 2, target
# CHECK-BE: bnu+ 0, target                  # encoding: [0x40,0xe3,A,0bAAAAAA00]
# CHECK-LE: bnu+ 0, target                  # encoding: [0bAAAAAA00,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns+ target
# CHECK-BE: bnua+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA10]
# CHECK-LE: bnua+ 2, target                 # encoding: [0bAAAAAA10,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa+ 2, target
# CHECK-BE: bnua+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA10]
# CHECK-LE: bnua+ 0, target                 # encoding: [0bAAAAAA10,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa+ target
# CHECK-BE: bnulr+ 2                        # encoding: [0x4c,0xeb,0x00,0x20]
# CHECK-LE: bnulr+ 2                        # encoding: [0x20,0x00,0xeb,0x4c]
            bnslr+ 2
# CHECK-BE: bnulr+ 0                        # encoding: [0x4c,0xe3,0x00,0x20]
# CHECK-LE: bnulr+ 0                        # encoding: [0x20,0x00,0xe3,0x4c]
            bnslr+
# CHECK-BE: bnuctr+ 2                       # encoding: [0x4c,0xeb,0x04,0x20]
# CHECK-LE: bnuctr+ 2                       # encoding: [0x20,0x04,0xeb,0x4c]
            bnsctr+ 2
# CHECK-BE: bnuctr+ 0                       # encoding: [0x4c,0xe3,0x04,0x20]
# CHECK-LE: bnuctr+ 0                       # encoding: [0x20,0x04,0xe3,0x4c]
            bnsctr+
# CHECK-BE: bnul+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA01]
# CHECK-LE: bnul+ 2, target                 # encoding: [0bAAAAAA01,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl+ 2, target
# CHECK-BE: bnul+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA01]
# CHECK-LE: bnul+ 0, target                 # encoding: [0bAAAAAA01,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl+ target
# CHECK-BE: bnula+ 2, target                # encoding: [0x40,0xeb,A,0bAAAAAA11]
# CHECK-LE: bnula+ 2, target                # encoding: [0bAAAAAA11,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla+ 2, target
# CHECK-BE: bnula+ 0, target                # encoding: [0x40,0xe3,A,0bAAAAAA11]
# CHECK-LE: bnula+ 0, target                # encoding: [0bAAAAAA11,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla+ target
# CHECK-BE: bnulrl+ 2                       # encoding: [0x4c,0xeb,0x00,0x21]
# CHECK-LE: bnulrl+ 2                       # encoding: [0x21,0x00,0xeb,0x4c]
            bnslrl+ 2
# CHECK-BE: bnulrl+ 0                       # encoding: [0x4c,0xe3,0x00,0x21]
# CHECK-LE: bnulrl+ 0                       # encoding: [0x21,0x00,0xe3,0x4c]
            bnslrl+
# CHECK-BE: bnuctrl+ 2                      # encoding: [0x4c,0xeb,0x04,0x21]
# CHECK-LE: bnuctrl+ 2                      # encoding: [0x21,0x04,0xeb,0x4c]
            bnsctrl+ 2
# CHECK-BE: bnuctrl+ 0                      # encoding: [0x4c,0xe3,0x04,0x21]
# CHECK-LE: bnuctrl+ 0                      # encoding: [0x21,0x04,0xe3,0x4c]
            bnsctrl+

# CHECK-BE: bnu- 2, target                  # encoding: [0x40,0xcb,A,0bAAAAAA00]
# CHECK-LE: bnu- 2, target                  # encoding: [0bAAAAAA00,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns- 2, target
# CHECK-BE: bnu- 0, target                  # encoding: [0x40,0xc3,A,0bAAAAAA00]
# CHECK-LE: bnu- 0, target                  # encoding: [0bAAAAAA00,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns- target
# CHECK-BE: bnua- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA10]
# CHECK-LE: bnua- 2, target                 # encoding: [0bAAAAAA10,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa- 2, target
# CHECK-BE: bnua- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA10]
# CHECK-LE: bnua- 0, target                 # encoding: [0bAAAAAA10,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa- target
# CHECK-BE: bnulr- 2                        # encoding: [0x4c,0xcb,0x00,0x20]
# CHECK-LE: bnulr- 2                        # encoding: [0x20,0x00,0xcb,0x4c]
            bnslr- 2
# CHECK-BE: bnulr- 0                        # encoding: [0x4c,0xc3,0x00,0x20]
# CHECK-LE: bnulr- 0                        # encoding: [0x20,0x00,0xc3,0x4c]
            bnslr-
# CHECK-BE: bnuctr- 2                       # encoding: [0x4c,0xcb,0x04,0x20]
# CHECK-LE: bnuctr- 2                       # encoding: [0x20,0x04,0xcb,0x4c]
            bnsctr- 2
# CHECK-BE: bnuctr- 0                       # encoding: [0x4c,0xc3,0x04,0x20]
# CHECK-LE: bnuctr- 0                       # encoding: [0x20,0x04,0xc3,0x4c]
            bnsctr-
# CHECK-BE: bnul- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA01]
# CHECK-LE: bnul- 2, target                 # encoding: [0bAAAAAA01,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl- 2, target
# CHECK-BE: bnul- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA01]
# CHECK-LE: bnul- 0, target                 # encoding: [0bAAAAAA01,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl- target
# CHECK-BE: bnula- 2, target                # encoding: [0x40,0xcb,A,0bAAAAAA11]
# CHECK-LE: bnula- 2, target                # encoding: [0bAAAAAA11,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla- 2, target
# CHECK-BE: bnula- 0, target                # encoding: [0x40,0xc3,A,0bAAAAAA11]
# CHECK-LE: bnula- 0, target                # encoding: [0bAAAAAA11,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla- target
# CHECK-BE: bnulrl- 2                       # encoding: [0x4c,0xcb,0x00,0x21]
# CHECK-LE: bnulrl- 2                       # encoding: [0x21,0x00,0xcb,0x4c]
            bnslrl- 2
# CHECK-BE: bnulrl- 0                       # encoding: [0x4c,0xc3,0x00,0x21]
# CHECK-LE: bnulrl- 0                       # encoding: [0x21,0x00,0xc3,0x4c]
            bnslrl-
# CHECK-BE: bnuctrl- 2                      # encoding: [0x4c,0xcb,0x04,0x21]
# CHECK-LE: bnuctrl- 2                      # encoding: [0x21,0x04,0xcb,0x4c]
            bnsctrl- 2
# CHECK-BE: bnuctrl- 0                      # encoding: [0x4c,0xc3,0x04,0x21]
# CHECK-LE: bnuctrl- 0                      # encoding: [0x21,0x04,0xc3,0x4c]
            bnsctrl-

# CHECK-BE: bun 2, target                   # encoding: [0x41,0x8b,A,0bAAAAAA00]
# CHECK-LE: bun 2, target                   # encoding: [0bAAAAAA00,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun 2, target
# CHECK-BE: bun 0, target                   # encoding: [0x41,0x83,A,0bAAAAAA00]
# CHECK-LE: bun 0, target                   # encoding: [0bAAAAAA00,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun target
# CHECK-BE: buna 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA10]
# CHECK-LE: buna 2, target                  # encoding: [0bAAAAAA10,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna 2, target
# CHECK-BE: buna 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA10]
# CHECK-LE: buna 0, target                  # encoding: [0bAAAAAA10,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna target
# CHECK-BE: bunlr 2                         # encoding: [0x4d,0x8b,0x00,0x20]
# CHECK-LE: bunlr 2                         # encoding: [0x20,0x00,0x8b,0x4d]
            bunlr 2
# CHECK-BE: bunlr 0                         # encoding: [0x4d,0x83,0x00,0x20]
# CHECK-LE: bunlr 0                         # encoding: [0x20,0x00,0x83,0x4d]
            bunlr
# CHECK-BE: bunctr 2                        # encoding: [0x4d,0x8b,0x04,0x20]
# CHECK-LE: bunctr 2                        # encoding: [0x20,0x04,0x8b,0x4d]
            bunctr 2
# CHECK-BE: bunctr 0                        # encoding: [0x4d,0x83,0x04,0x20]
# CHECK-LE: bunctr 0                        # encoding: [0x20,0x04,0x83,0x4d]
            bunctr
# CHECK-BE: bunl 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA01]
# CHECK-LE: bunl 2, target                  # encoding: [0bAAAAAA01,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl 2, target
# CHECK-BE: bunl 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA01]
# CHECK-LE: bunl 0, target                  # encoding: [0bAAAAAA01,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl target
# CHECK-BE: bunla 2, target                 # encoding: [0x41,0x8b,A,0bAAAAAA11]
# CHECK-LE: bunla 2, target                 # encoding: [0bAAAAAA11,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla 2, target
# CHECK-BE: bunla 0, target                 # encoding: [0x41,0x83,A,0bAAAAAA11]
# CHECK-LE: bunla 0, target                 # encoding: [0bAAAAAA11,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla target
# CHECK-BE: bunlrl 2                        # encoding: [0x4d,0x8b,0x00,0x21]
# CHECK-LE: bunlrl 2                        # encoding: [0x21,0x00,0x8b,0x4d]
            bunlrl 2
# CHECK-BE: bunlrl 0                        # encoding: [0x4d,0x83,0x00,0x21]
# CHECK-LE: bunlrl 0                        # encoding: [0x21,0x00,0x83,0x4d]
            bunlrl
# CHECK-BE: bunctrl 2                       # encoding: [0x4d,0x8b,0x04,0x21]
# CHECK-LE: bunctrl 2                       # encoding: [0x21,0x04,0x8b,0x4d]
            bunctrl 2
# CHECK-BE: bunctrl 0                       # encoding: [0x4d,0x83,0x04,0x21]
# CHECK-LE: bunctrl 0                       # encoding: [0x21,0x04,0x83,0x4d]
            bunctrl

# CHECK-BE: bun+ 2, target                  # encoding: [0x41,0xeb,A,0bAAAAAA00]
# CHECK-LE: bun+ 2, target                  # encoding: [0bAAAAAA00,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun+ 2, target
# CHECK-BE: bun+ 0, target                  # encoding: [0x41,0xe3,A,0bAAAAAA00]
# CHECK-LE: bun+ 0, target                  # encoding: [0bAAAAAA00,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun+ target
# CHECK-BE: buna+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA10]
# CHECK-LE: buna+ 2, target                 # encoding: [0bAAAAAA10,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna+ 2, target
# CHECK-BE: buna+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA10]
# CHECK-LE: buna+ 0, target                 # encoding: [0bAAAAAA10,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna+ target
# CHECK-BE: bunlr+ 2                        # encoding: [0x4d,0xeb,0x00,0x20]
# CHECK-LE: bunlr+ 2                        # encoding: [0x20,0x00,0xeb,0x4d]
            bunlr+ 2
# CHECK-BE: bunlr+ 0                        # encoding: [0x4d,0xe3,0x00,0x20]
# CHECK-LE: bunlr+ 0                        # encoding: [0x20,0x00,0xe3,0x4d]
            bunlr+
# CHECK-BE: bunctr+ 2                       # encoding: [0x4d,0xeb,0x04,0x20]
# CHECK-LE: bunctr+ 2                       # encoding: [0x20,0x04,0xeb,0x4d]
            bunctr+ 2
# CHECK-BE: bunctr+ 0                       # encoding: [0x4d,0xe3,0x04,0x20]
# CHECK-LE: bunctr+ 0                       # encoding: [0x20,0x04,0xe3,0x4d]
            bunctr+
# CHECK-BE: bunl+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA01]
# CHECK-LE: bunl+ 2, target                 # encoding: [0bAAAAAA01,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl+ 2, target
# CHECK-BE: bunl+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA01]
# CHECK-LE: bunl+ 0, target                 # encoding: [0bAAAAAA01,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl+ target
# CHECK-BE: bunla+ 2, target                # encoding: [0x41,0xeb,A,0bAAAAAA11]
# CHECK-LE: bunla+ 2, target                # encoding: [0bAAAAAA11,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla+ 2, target
# CHECK-BE: bunla+ 0, target                # encoding: [0x41,0xe3,A,0bAAAAAA11]
# CHECK-LE: bunla+ 0, target                # encoding: [0bAAAAAA11,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla+ target
# CHECK-BE: bunlrl+ 2                       # encoding: [0x4d,0xeb,0x00,0x21]
# CHECK-LE: bunlrl+ 2                       # encoding: [0x21,0x00,0xeb,0x4d]
            bunlrl+ 2
# CHECK-BE: bunlrl+ 0                       # encoding: [0x4d,0xe3,0x00,0x21]
# CHECK-LE: bunlrl+ 0                       # encoding: [0x21,0x00,0xe3,0x4d]
            bunlrl+
# CHECK-BE: bunctrl+ 2                      # encoding: [0x4d,0xeb,0x04,0x21]
# CHECK-LE: bunctrl+ 2                      # encoding: [0x21,0x04,0xeb,0x4d]
            bunctrl+ 2
# CHECK-BE: bunctrl+ 0                      # encoding: [0x4d,0xe3,0x04,0x21]
# CHECK-LE: bunctrl+ 0                      # encoding: [0x21,0x04,0xe3,0x4d]
            bunctrl+

# CHECK-BE: bun- 2, target                  # encoding: [0x41,0xcb,A,0bAAAAAA00]
# CHECK-LE: bun- 2, target                  # encoding: [0bAAAAAA00,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun- 2, target
# CHECK-BE: bun- 0, target                  # encoding: [0x41,0xc3,A,0bAAAAAA00]
# CHECK-LE: bun- 0, target                  # encoding: [0bAAAAAA00,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun- target
# CHECK-BE: buna- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA10]
# CHECK-LE: buna- 2, target                 # encoding: [0bAAAAAA10,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna- 2, target
# CHECK-BE: buna- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA10]
# CHECK-LE: buna- 0, target                 # encoding: [0bAAAAAA10,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna- target
# CHECK-BE: bunlr- 2                        # encoding: [0x4d,0xcb,0x00,0x20]
# CHECK-LE: bunlr- 2                        # encoding: [0x20,0x00,0xcb,0x4d]
            bunlr- 2
# CHECK-BE: bunlr- 0                        # encoding: [0x4d,0xc3,0x00,0x20]
# CHECK-LE: bunlr- 0                        # encoding: [0x20,0x00,0xc3,0x4d]
            bunlr-
# CHECK-BE: bunctr- 2                       # encoding: [0x4d,0xcb,0x04,0x20]
# CHECK-LE: bunctr- 2                       # encoding: [0x20,0x04,0xcb,0x4d]
            bunctr- 2
# CHECK-BE: bunctr- 0                       # encoding: [0x4d,0xc3,0x04,0x20]
# CHECK-LE: bunctr- 0                       # encoding: [0x20,0x04,0xc3,0x4d]
            bunctr-
# CHECK-BE: bunl- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA01]
# CHECK-LE: bunl- 2, target                 # encoding: [0bAAAAAA01,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl- 2, target
# CHECK-BE: bunl- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA01]
# CHECK-LE: bunl- 0, target                 # encoding: [0bAAAAAA01,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl- target
# CHECK-BE: bunla- 2, target                # encoding: [0x41,0xcb,A,0bAAAAAA11]
# CHECK-LE: bunla- 2, target                # encoding: [0bAAAAAA11,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla- 2, target
# CHECK-BE: bunla- 0, target                # encoding: [0x41,0xc3,A,0bAAAAAA11]
# CHECK-LE: bunla- 0, target                # encoding: [0bAAAAAA11,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla- target
# CHECK-BE: bunlrl- 2                       # encoding: [0x4d,0xcb,0x00,0x21]
# CHECK-LE: bunlrl- 2                       # encoding: [0x21,0x00,0xcb,0x4d]
            bunlrl- 2
# CHECK-BE: bunlrl- 0                       # encoding: [0x4d,0xc3,0x00,0x21]
# CHECK-LE: bunlrl- 0                       # encoding: [0x21,0x00,0xc3,0x4d]
            bunlrl-
# CHECK-BE: bunctrl- 2                      # encoding: [0x4d,0xcb,0x04,0x21]
# CHECK-LE: bunctrl- 2                      # encoding: [0x21,0x04,0xcb,0x4d]
            bunctrl- 2
# CHECK-BE: bunctrl- 0                      # encoding: [0x4d,0xc3,0x04,0x21]
# CHECK-LE: bunctrl- 0                      # encoding: [0x21,0x04,0xc3,0x4d]
            bunctrl-

# CHECK-BE: bnu 2, target                   # encoding: [0x40,0x8b,A,0bAAAAAA00]
# CHECK-LE: bnu 2, target                   # encoding: [0bAAAAAA00,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu 2, target
# CHECK-BE: bnu 0, target                   # encoding: [0x40,0x83,A,0bAAAAAA00]
# CHECK-LE: bnu 0, target                   # encoding: [0bAAAAAA00,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu target
# CHECK-BE: bnua 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA10]
# CHECK-LE: bnua 2, target                  # encoding: [0bAAAAAA10,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua 2, target
# CHECK-BE: bnua 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA10]
# CHECK-LE: bnua 0, target                  # encoding: [0bAAAAAA10,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua target
# CHECK-BE: bnulr 2                         # encoding: [0x4c,0x8b,0x00,0x20]
# CHECK-LE: bnulr 2                         # encoding: [0x20,0x00,0x8b,0x4c]
            bnulr 2
# CHECK-BE: bnulr 0                         # encoding: [0x4c,0x83,0x00,0x20]
# CHECK-LE: bnulr 0                         # encoding: [0x20,0x00,0x83,0x4c]
            bnulr
# CHECK-BE: bnuctr 2                        # encoding: [0x4c,0x8b,0x04,0x20]
# CHECK-LE: bnuctr 2                        # encoding: [0x20,0x04,0x8b,0x4c]
            bnuctr 2
# CHECK-BE: bnuctr 0                        # encoding: [0x4c,0x83,0x04,0x20]
# CHECK-LE: bnuctr 0                        # encoding: [0x20,0x04,0x83,0x4c]
            bnuctr
# CHECK-BE: bnul 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA01]
# CHECK-LE: bnul 2, target                  # encoding: [0bAAAAAA01,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul 2, target
# CHECK-BE: bnul 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA01]
# CHECK-LE: bnul 0, target                  # encoding: [0bAAAAAA01,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul target
# CHECK-BE: bnula 2, target                 # encoding: [0x40,0x8b,A,0bAAAAAA11]
# CHECK-LE: bnula 2, target                 # encoding: [0bAAAAAA11,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula 2, target
# CHECK-BE: bnula 0, target                 # encoding: [0x40,0x83,A,0bAAAAAA11]
# CHECK-LE: bnula 0, target                 # encoding: [0bAAAAAA11,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula target
# CHECK-BE: bnulrl 2                        # encoding: [0x4c,0x8b,0x00,0x21]
# CHECK-LE: bnulrl 2                        # encoding: [0x21,0x00,0x8b,0x4c]
            bnulrl 2
# CHECK-BE: bnulrl 0                        # encoding: [0x4c,0x83,0x00,0x21]
# CHECK-LE: bnulrl 0                        # encoding: [0x21,0x00,0x83,0x4c]
            bnulrl
# CHECK-BE: bnuctrl 2                       # encoding: [0x4c,0x8b,0x04,0x21]
# CHECK-LE: bnuctrl 2                       # encoding: [0x21,0x04,0x8b,0x4c]
            bnuctrl 2
# CHECK-BE: bnuctrl 0                       # encoding: [0x4c,0x83,0x04,0x21]
# CHECK-LE: bnuctrl 0                       # encoding: [0x21,0x04,0x83,0x4c]
            bnuctrl

# CHECK-BE: bnu+ 2, target                  # encoding: [0x40,0xeb,A,0bAAAAAA00]
# CHECK-LE: bnu+ 2, target                  # encoding: [0bAAAAAA00,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu+ 2, target
# CHECK-BE: bnu+ 0, target                  # encoding: [0x40,0xe3,A,0bAAAAAA00]
# CHECK-LE: bnu+ 0, target                  # encoding: [0bAAAAAA00,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu+ target
# CHECK-BE: bnua+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA10]
# CHECK-LE: bnua+ 2, target                 # encoding: [0bAAAAAA10,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua+ 2, target
# CHECK-BE: bnua+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA10]
# CHECK-LE: bnua+ 0, target                 # encoding: [0bAAAAAA10,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua+ target
# CHECK-BE: bnulr+ 2                        # encoding: [0x4c,0xeb,0x00,0x20]
# CHECK-LE: bnulr+ 2                        # encoding: [0x20,0x00,0xeb,0x4c]
            bnulr+ 2
# CHECK-BE: bnulr+ 0                        # encoding: [0x4c,0xe3,0x00,0x20]
# CHECK-LE: bnulr+ 0                        # encoding: [0x20,0x00,0xe3,0x4c]
            bnulr+
# CHECK-BE: bnuctr+ 2                       # encoding: [0x4c,0xeb,0x04,0x20]
# CHECK-LE: bnuctr+ 2                       # encoding: [0x20,0x04,0xeb,0x4c]
            bnuctr+ 2
# CHECK-BE: bnuctr+ 0                       # encoding: [0x4c,0xe3,0x04,0x20]
# CHECK-LE: bnuctr+ 0                       # encoding: [0x20,0x04,0xe3,0x4c]
            bnuctr+
# CHECK-BE: bnul+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA01]
# CHECK-LE: bnul+ 2, target                 # encoding: [0bAAAAAA01,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul+ 2, target
# CHECK-BE: bnul+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA01]
# CHECK-LE: bnul+ 0, target                 # encoding: [0bAAAAAA01,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul+ target
# CHECK-BE: bnula+ 2, target                # encoding: [0x40,0xeb,A,0bAAAAAA11]
# CHECK-LE: bnula+ 2, target                # encoding: [0bAAAAAA11,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula+ 2, target
# CHECK-BE: bnula+ 0, target                # encoding: [0x40,0xe3,A,0bAAAAAA11]
# CHECK-LE: bnula+ 0, target                # encoding: [0bAAAAAA11,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula+ target
# CHECK-BE: bnulrl+ 2                       # encoding: [0x4c,0xeb,0x00,0x21]
# CHECK-LE: bnulrl+ 2                       # encoding: [0x21,0x00,0xeb,0x4c]
            bnulrl+ 2
# CHECK-BE: bnulrl+ 0                       # encoding: [0x4c,0xe3,0x00,0x21]
# CHECK-LE: bnulrl+ 0                       # encoding: [0x21,0x00,0xe3,0x4c]
            bnulrl+
# CHECK-BE: bnuctrl+ 2                      # encoding: [0x4c,0xeb,0x04,0x21]
# CHECK-LE: bnuctrl+ 2                      # encoding: [0x21,0x04,0xeb,0x4c]
            bnuctrl+ 2
# CHECK-BE: bnuctrl+ 0                      # encoding: [0x4c,0xe3,0x04,0x21]
# CHECK-LE: bnuctrl+ 0                      # encoding: [0x21,0x04,0xe3,0x4c]
            bnuctrl+

# CHECK-BE: bnu- 2, target                  # encoding: [0x40,0xcb,A,0bAAAAAA00]
# CHECK-LE: bnu- 2, target                  # encoding: [0bAAAAAA00,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu- 2, target
# CHECK-BE: bnu- 0, target                  # encoding: [0x40,0xc3,A,0bAAAAAA00]
# CHECK-LE: bnu- 0, target                  # encoding: [0bAAAAAA00,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu- target
# CHECK-BE: bnua- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA10]
# CHECK-LE: bnua- 2, target                 # encoding: [0bAAAAAA10,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua- 2, target
# CHECK-BE: bnua- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA10]
# CHECK-LE: bnua- 0, target                 # encoding: [0bAAAAAA10,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua- target
# CHECK-BE: bnulr- 2                        # encoding: [0x4c,0xcb,0x00,0x20]
# CHECK-LE: bnulr- 2                        # encoding: [0x20,0x00,0xcb,0x4c]
            bnulr- 2
# CHECK-BE: bnulr- 0                        # encoding: [0x4c,0xc3,0x00,0x20]
# CHECK-LE: bnulr- 0                        # encoding: [0x20,0x00,0xc3,0x4c]
            bnulr-
# CHECK-BE: bnuctr- 2                       # encoding: [0x4c,0xcb,0x04,0x20]
# CHECK-LE: bnuctr- 2                       # encoding: [0x20,0x04,0xcb,0x4c]
            bnuctr- 2
# CHECK-BE: bnuctr- 0                       # encoding: [0x4c,0xc3,0x04,0x20]
# CHECK-LE: bnuctr- 0                       # encoding: [0x20,0x04,0xc3,0x4c]
            bnuctr-
# CHECK-BE: bnul- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA01]
# CHECK-LE: bnul- 2, target                 # encoding: [0bAAAAAA01,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul- 2, target
# CHECK-BE: bnul- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA01]
# CHECK-LE: bnul- 0, target                 # encoding: [0bAAAAAA01,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul- target
# CHECK-BE: bnula- 2, target                # encoding: [0x40,0xcb,A,0bAAAAAA11]
# CHECK-LE: bnula- 2, target                # encoding: [0bAAAAAA11,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula- 2, target
# CHECK-BE: bnula- 0, target                # encoding: [0x40,0xc3,A,0bAAAAAA11]
# CHECK-LE: bnula- 0, target                # encoding: [0bAAAAAA11,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula- target
# CHECK-BE: bnulrl- 2                       # encoding: [0x4c,0xcb,0x00,0x21]
# CHECK-LE: bnulrl- 2                       # encoding: [0x21,0x00,0xcb,0x4c]
            bnulrl- 2
# CHECK-BE: bnulrl- 0                       # encoding: [0x4c,0xc3,0x00,0x21]
# CHECK-LE: bnulrl- 0                       # encoding: [0x21,0x00,0xc3,0x4c]
            bnulrl-
# CHECK-BE: bnuctrl- 2                      # encoding: [0x4c,0xcb,0x04,0x21]
# CHECK-LE: bnuctrl- 2                      # encoding: [0x21,0x04,0xcb,0x4c]
            bnuctrl- 2
# CHECK-BE: bnuctrl- 0                      # encoding: [0x4c,0xc3,0x04,0x21]
# CHECK-LE: bnuctrl- 0                      # encoding: [0x21,0x04,0xc3,0x4c]
            bnuctrl-

# Condition register logical mnemonics

# CHECK-BE: creqv 2, 2, 2                   # encoding: [0x4c,0x42,0x12,0x42]
# CHECK-LE: creqv 2, 2, 2                   # encoding: [0x42,0x12,0x42,0x4c]
            crset 2
# CHECK-BE: crxor 2, 2, 2                   # encoding: [0x4c,0x42,0x11,0x82]
# CHECK-LE: crxor 2, 2, 2                   # encoding: [0x82,0x11,0x42,0x4c]
            crclr 2
# CHECK-BE: cror 2, 3, 3                    # encoding: [0x4c,0x43,0x1b,0x82]
# CHECK-LE: cror 2, 3, 3                    # encoding: [0x82,0x1b,0x43,0x4c]
            crmove 2, 3
# CHECK-BE: crnor 2, 3, 3                   # encoding: [0x4c,0x43,0x18,0x42]
# CHECK-LE: crnor 2, 3, 3                   # encoding: [0x42,0x18,0x43,0x4c]
            crnot 2, 3

# Subtract mnemonics

# CHECK-BE: addi 2, 3, -128                 # encoding: [0x38,0x43,0xff,0x80]
# CHECK-LE: addi 2, 3, -128                 # encoding: [0x80,0xff,0x43,0x38]
            subi 2, 3, 128
# CHECK-BE: addis 2, 3, -128                # encoding: [0x3c,0x43,0xff,0x80]
# CHECK-LE: addis 2, 3, -128                # encoding: [0x80,0xff,0x43,0x3c]
            subis 2, 3, 128
# CHECK-BE: addic 2, 3, -128                # encoding: [0x30,0x43,0xff,0x80]
# CHECK-LE: addic 2, 3, -128                # encoding: [0x80,0xff,0x43,0x30]
            subic 2, 3, 128
# CHECK-BE: addic. 2, 3, -128               # encoding: [0x34,0x43,0xff,0x80]
# CHECK-LE: addic. 2, 3, -128               # encoding: [0x80,0xff,0x43,0x34]
            subic. 2, 3, 128

# CHECK-BE: subf 2, 4, 3                    # encoding: [0x7c,0x44,0x18,0x50]
# CHECK-LE: subf 2, 4, 3                    # encoding: [0x50,0x18,0x44,0x7c]
            sub 2, 3, 4
# CHECK-BE: subf. 2, 4, 3                   # encoding: [0x7c,0x44,0x18,0x51]
# CHECK-LE: subf. 2, 4, 3                   # encoding: [0x51,0x18,0x44,0x7c]
            sub. 2, 3, 4
# CHECK-BE: subfc 2, 4, 3                   # encoding: [0x7c,0x44,0x18,0x10]
# CHECK-LE: subfc 2, 4, 3                   # encoding: [0x10,0x18,0x44,0x7c]
            subc 2, 3, 4
# CHECK-BE: subfc. 2, 4, 3                  # encoding: [0x7c,0x44,0x18,0x11]
# CHECK-LE: subfc. 2, 4, 3                  # encoding: [0x11,0x18,0x44,0x7c]
            subc. 2, 3, 4

# Compare mnemonics

# CHECK-BE: cmpdi 2, 3, 128                 # encoding: [0x2d,0x23,0x00,0x80]
# CHECK-LE: cmpdi 2, 3, 128                 # encoding: [0x80,0x00,0x23,0x2d]
            cmpdi 2, 3, 128
# CHECK-BE: cmpdi 0, 3, 128                 # encoding: [0x2c,0x23,0x00,0x80]
# CHECK-LE: cmpdi 0, 3, 128                 # encoding: [0x80,0x00,0x23,0x2c]
            cmpdi 3, 128
# CHECK-BE: cmpd 2, 3, 4                    # encoding: [0x7d,0x23,0x20,0x00]
# CHECK-LE: cmpd 2, 3, 4                    # encoding: [0x00,0x20,0x23,0x7d]
            cmpd 2, 3, 4
# CHECK-BE: cmpd 0, 3, 4                    # encoding: [0x7c,0x23,0x20,0x00]
# CHECK-LE: cmpd 0, 3, 4                    # encoding: [0x00,0x20,0x23,0x7c]
            cmpd 3, 4
# CHECK-BE: cmpldi 2, 3, 128                # encoding: [0x29,0x23,0x00,0x80]
# CHECK-LE: cmpldi 2, 3, 128                # encoding: [0x80,0x00,0x23,0x29]
            cmpldi 2, 3, 128
# CHECK-BE: cmpldi 0, 3, 128                # encoding: [0x28,0x23,0x00,0x80]
# CHECK-LE: cmpldi 0, 3, 128                # encoding: [0x80,0x00,0x23,0x28]
            cmpldi 3, 128
# CHECK-BE: cmpld 2, 3, 4                   # encoding: [0x7d,0x23,0x20,0x40]
# CHECK-LE: cmpld 2, 3, 4                   # encoding: [0x40,0x20,0x23,0x7d]
            cmpld 2, 3, 4
# CHECK-BE: cmpld 0, 3, 4                   # encoding: [0x7c,0x23,0x20,0x40]
# CHECK-LE: cmpld 0, 3, 4                   # encoding: [0x40,0x20,0x23,0x7c]
            cmpld 3, 4

# CHECK-BE: cmpwi 2, 3, 128                 # encoding: [0x2d,0x03,0x00,0x80]
# CHECK-LE: cmpwi 2, 3, 128                 # encoding: [0x80,0x00,0x03,0x2d]
            cmpwi 2, 3, 128
# CHECK-BE: cmpwi 0, 3, 128                 # encoding: [0x2c,0x03,0x00,0x80]
# CHECK-LE: cmpwi 0, 3, 128                 # encoding: [0x80,0x00,0x03,0x2c]
            cmpwi 3, 128
# CHECK-BE: cmpw 2, 3, 4                    # encoding: [0x7d,0x03,0x20,0x00]
# CHECK-LE: cmpw 2, 3, 4                    # encoding: [0x00,0x20,0x03,0x7d]
            cmpw 2, 3, 4
# CHECK-BE: cmpw 0, 3, 4                    # encoding: [0x7c,0x03,0x20,0x00]
# CHECK-LE: cmpw 0, 3, 4                    # encoding: [0x00,0x20,0x03,0x7c]
            cmpw 3, 4
# CHECK-BE: cmplwi 2, 3, 128                # encoding: [0x29,0x03,0x00,0x80]
# CHECK-LE: cmplwi 2, 3, 128                # encoding: [0x80,0x00,0x03,0x29]
            cmplwi 2, 3, 128
# CHECK-BE: cmplwi 0, 3, 128                # encoding: [0x28,0x03,0x00,0x80]
# CHECK-LE: cmplwi 0, 3, 128                # encoding: [0x80,0x00,0x03,0x28]
            cmplwi 3, 128
# CHECK-BE: cmplw 2, 3, 4                   # encoding: [0x7d,0x03,0x20,0x40]
# CHECK-LE: cmplw 2, 3, 4                   # encoding: [0x40,0x20,0x03,0x7d]
            cmplw 2, 3, 4
# CHECK-BE: cmplw 0, 3, 4                   # encoding: [0x7c,0x03,0x20,0x40]
# CHECK-LE: cmplw 0, 3, 4                   # encoding: [0x40,0x20,0x03,0x7c]
            cmplw 3, 4

# Trap mnemonics

# CHECK-BE: twi 16, 3, 4                    # encoding: [0x0e,0x03,0x00,0x04]
# CHECK-LE: twi 16, 3, 4                    # encoding: [0x04,0x00,0x03,0x0e]
            twlti 3, 4
# CHECK-BE: tw 16, 3, 4                     # encoding: [0x7e,0x03,0x20,0x08]
# CHECK-LE: tw 16, 3, 4                     # encoding: [0x08,0x20,0x03,0x7e]
            twlt 3, 4
# CHECK-BE: tdi 16, 3, 4                    # encoding: [0x0a,0x03,0x00,0x04]
# CHECK-LE: tdi 16, 3, 4                    # encoding: [0x04,0x00,0x03,0x0a]
            tdlti 3, 4
# CHECK-BE: td 16, 3, 4                     # encoding: [0x7e,0x03,0x20,0x88]
# CHECK-LE: td 16, 3, 4                     # encoding: [0x88,0x20,0x03,0x7e]
            tdlt 3, 4

# CHECK-BE: twi 20, 3, 4                    # encoding: [0x0e,0x83,0x00,0x04]
# CHECK-LE: twi 20, 3, 4                    # encoding: [0x04,0x00,0x83,0x0e]
            twlei 3, 4
# CHECK-BE: tw 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x08]
# CHECK-LE: tw 20, 3, 4                     # encoding: [0x08,0x20,0x83,0x7e]
            twle 3, 4
# CHECK-BE: tdi 20, 3, 4                    # encoding: [0x0a,0x83,0x00,0x04]
# CHECK-LE: tdi 20, 3, 4                    # encoding: [0x04,0x00,0x83,0x0a]
            tdlei 3, 4
# CHECK-BE: td 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x88]
# CHECK-LE: td 20, 3, 4                     # encoding: [0x88,0x20,0x83,0x7e]
            tdle 3, 4

# CHECK-BE: twi 4, 3, 4                     # encoding: [0x0c,0x83,0x00,0x04]
# CHECK-LE: twi 4, 3, 4                     # encoding: [0x04,0x00,0x83,0x0c]
            tweqi 3, 4
# CHECK-BE: tw 4, 3, 4                      # encoding: [0x7c,0x83,0x20,0x08]
# CHECK-LE: tw 4, 3, 4                      # encoding: [0x08,0x20,0x83,0x7c]
            tweq 3, 4
# CHECK-BE: tdi 4, 3, 4                     # encoding: [0x08,0x83,0x00,0x04]
# CHECK-LE: tdi 4, 3, 4                     # encoding: [0x04,0x00,0x83,0x08]
            tdeqi 3, 4
# CHECK-BE: td 4, 3, 4                      # encoding: [0x7c,0x83,0x20,0x88]
# CHECK-LE: td 4, 3, 4                      # encoding: [0x88,0x20,0x83,0x7c]
            tdeq 3, 4

# CHECK-BE: twi 12, 3, 4                    # encoding: [0x0d,0x83,0x00,0x04]
# CHECK-LE: twi 12, 3, 4                    # encoding: [0x04,0x00,0x83,0x0d]
            twgei 3, 4
# CHECK-BE: tw 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x08]
# CHECK-LE: tw 12, 3, 4                     # encoding: [0x08,0x20,0x83,0x7d]
            twge 3, 4
# CHECK-BE: tdi 12, 3, 4                    # encoding: [0x09,0x83,0x00,0x04]
# CHECK-LE: tdi 12, 3, 4                    # encoding: [0x04,0x00,0x83,0x09]
            tdgei 3, 4
# CHECK-BE: td 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x88]
# CHECK-LE: td 12, 3, 4                     # encoding: [0x88,0x20,0x83,0x7d]
            tdge 3, 4

# CHECK-BE: twi 8, 3, 4                     # encoding: [0x0d,0x03,0x00,0x04]
# CHECK-LE: twi 8, 3, 4                     # encoding: [0x04,0x00,0x03,0x0d]
            twgti 3, 4
# CHECK-BE: tw 8, 3, 4                      # encoding: [0x7d,0x03,0x20,0x08]
# CHECK-LE: tw 8, 3, 4                      # encoding: [0x08,0x20,0x03,0x7d]
            twgt 3, 4
# CHECK-BE: tdi 8, 3, 4                     # encoding: [0x09,0x03,0x00,0x04]
# CHECK-LE: tdi 8, 3, 4                     # encoding: [0x04,0x00,0x03,0x09]
            tdgti 3, 4
# CHECK-BE: td 8, 3, 4                      # encoding: [0x7d,0x03,0x20,0x88]
# CHECK-LE: td 8, 3, 4                      # encoding: [0x88,0x20,0x03,0x7d]
            tdgt 3, 4

# CHECK-BE: twi 12, 3, 4                    # encoding: [0x0d,0x83,0x00,0x04]
# CHECK-LE: twi 12, 3, 4                    # encoding: [0x04,0x00,0x83,0x0d]
            twnli 3, 4
# CHECK-BE: tw 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x08]
# CHECK-LE: tw 12, 3, 4                     # encoding: [0x08,0x20,0x83,0x7d]
            twnl 3, 4
# CHECK-BE: tdi 12, 3, 4                    # encoding: [0x09,0x83,0x00,0x04]
# CHECK-LE: tdi 12, 3, 4                    # encoding: [0x04,0x00,0x83,0x09]
            tdnli 3, 4
# CHECK-BE: td 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x88]
# CHECK-LE: td 12, 3, 4                     # encoding: [0x88,0x20,0x83,0x7d]
            tdnl 3, 4

# CHECK-BE: twi 24, 3, 4                    # encoding: [0x0f,0x03,0x00,0x04]
# CHECK-LE: twi 24, 3, 4                    # encoding: [0x04,0x00,0x03,0x0f]
            twnei 3, 4
# CHECK-BE: tw 24, 3, 4                     # encoding: [0x7f,0x03,0x20,0x08]
# CHECK-LE: tw 24, 3, 4                     # encoding: [0x08,0x20,0x03,0x7f]
            twne 3, 4
# CHECK-BE: tdi 24, 3, 4                    # encoding: [0x0b,0x03,0x00,0x04]
# CHECK-LE: tdi 24, 3, 4                    # encoding: [0x04,0x00,0x03,0x0b]
            tdnei 3, 4
# CHECK-BE: td 24, 3, 4                     # encoding: [0x7f,0x03,0x20,0x88]
# CHECK-LE: td 24, 3, 4                     # encoding: [0x88,0x20,0x03,0x7f]
            tdne 3, 4

# CHECK-BE: twi 20, 3, 4                    # encoding: [0x0e,0x83,0x00,0x04]
# CHECK-LE: twi 20, 3, 4                    # encoding: [0x04,0x00,0x83,0x0e]
            twngi 3, 4
# CHECK-BE: tw 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x08]
# CHECK-LE: tw 20, 3, 4                     # encoding: [0x08,0x20,0x83,0x7e]
            twng 3, 4
# CHECK-BE: tdi 20, 3, 4                    # encoding: [0x0a,0x83,0x00,0x04]
# CHECK-LE: tdi 20, 3, 4                    # encoding: [0x04,0x00,0x83,0x0a]
            tdngi 3, 4
# CHECK-BE: td 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x88]
# CHECK-LE: td 20, 3, 4                     # encoding: [0x88,0x20,0x83,0x7e]
            tdng 3, 4

# CHECK-BE: twi 2, 3, 4                     # encoding: [0x0c,0x43,0x00,0x04]
# CHECK-LE: twi 2, 3, 4                     # encoding: [0x04,0x00,0x43,0x0c]
            twllti 3, 4
# CHECK-BE: tw 2, 3, 4                      # encoding: [0x7c,0x43,0x20,0x08]
# CHECK-LE: tw 2, 3, 4                      # encoding: [0x08,0x20,0x43,0x7c]
            twllt 3, 4
# CHECK-BE: tdi 2, 3, 4                     # encoding: [0x08,0x43,0x00,0x04]
# CHECK-LE: tdi 2, 3, 4                     # encoding: [0x04,0x00,0x43,0x08]
            tdllti 3, 4
# CHECK-BE: td 2, 3, 4                      # encoding: [0x7c,0x43,0x20,0x88]
# CHECK-LE: td 2, 3, 4                      # encoding: [0x88,0x20,0x43,0x7c]
            tdllt 3, 4

# CHECK-BE: twi 6, 3, 4                     # encoding: [0x0c,0xc3,0x00,0x04]
# CHECK-LE: twi 6, 3, 4                     # encoding: [0x04,0x00,0xc3,0x0c]
            twllei 3, 4
# CHECK-BE: tw 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x08]
# CHECK-LE: tw 6, 3, 4                      # encoding: [0x08,0x20,0xc3,0x7c]
            twlle 3, 4
# CHECK-BE: tdi 6, 3, 4                     # encoding: [0x08,0xc3,0x00,0x04]
# CHECK-LE: tdi 6, 3, 4                     # encoding: [0x04,0x00,0xc3,0x08]
            tdllei 3, 4
# CHECK-BE: td 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x88]
# CHECK-LE: td 6, 3, 4                      # encoding: [0x88,0x20,0xc3,0x7c]
            tdlle 3, 4

# CHECK-BE: twi 5, 3, 4                     # encoding: [0x0c,0xa3,0x00,0x04]
# CHECK-LE: twi 5, 3, 4                     # encoding: [0x04,0x00,0xa3,0x0c]
            twlgei 3, 4
# CHECK-BE: tw 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x08]
# CHECK-LE: tw 5, 3, 4                      # encoding: [0x08,0x20,0xa3,0x7c]
            twlge 3, 4
# CHECK-BE: tdi 5, 3, 4                     # encoding: [0x08,0xa3,0x00,0x04]
# CHECK-LE: tdi 5, 3, 4                     # encoding: [0x04,0x00,0xa3,0x08]
            tdlgei 3, 4
# CHECK-BE: td 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x88]
# CHECK-LE: td 5, 3, 4                      # encoding: [0x88,0x20,0xa3,0x7c]
            tdlge 3, 4

# CHECK-BE: twi 1, 3, 4                     # encoding: [0x0c,0x23,0x00,0x04]
# CHECK-LE: twi 1, 3, 4                     # encoding: [0x04,0x00,0x23,0x0c]
            twlgti 3, 4
# CHECK-BE: tw 1, 3, 4                      # encoding: [0x7c,0x23,0x20,0x08]
# CHECK-LE: tw 1, 3, 4                      # encoding: [0x08,0x20,0x23,0x7c]
            twlgt 3, 4
# CHECK-BE: tdi 1, 3, 4                     # encoding: [0x08,0x23,0x00,0x04]
# CHECK-LE: tdi 1, 3, 4                     # encoding: [0x04,0x00,0x23,0x08]
            tdlgti 3, 4
# CHECK-BE: td 1, 3, 4                      # encoding: [0x7c,0x23,0x20,0x88]
# CHECK-LE: td 1, 3, 4                      # encoding: [0x88,0x20,0x23,0x7c]
            tdlgt 3, 4

# CHECK-BE: twi 5, 3, 4                     # encoding: [0x0c,0xa3,0x00,0x04]
# CHECK-LE: twi 5, 3, 4                     # encoding: [0x04,0x00,0xa3,0x0c]
            twlnli 3, 4
# CHECK-BE: tw 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x08]
# CHECK-LE: tw 5, 3, 4                      # encoding: [0x08,0x20,0xa3,0x7c]
            twlnl 3, 4
# CHECK-BE: tdi 5, 3, 4                     # encoding: [0x08,0xa3,0x00,0x04]
# CHECK-LE: tdi 5, 3, 4                     # encoding: [0x04,0x00,0xa3,0x08]
            tdlnli 3, 4
# CHECK-BE: td 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x88]
# CHECK-LE: td 5, 3, 4                      # encoding: [0x88,0x20,0xa3,0x7c]
            tdlnl 3, 4

# CHECK-BE: twi 6, 3, 4                     # encoding: [0x0c,0xc3,0x00,0x04]
# CHECK-LE: twi 6, 3, 4                     # encoding: [0x04,0x00,0xc3,0x0c]
            twlngi 3, 4
# CHECK-BE: tw 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x08]
# CHECK-LE: tw 6, 3, 4                      # encoding: [0x08,0x20,0xc3,0x7c]
            twlng 3, 4
# CHECK-BE: tdi 6, 3, 4                     # encoding: [0x08,0xc3,0x00,0x04]
# CHECK-LE: tdi 6, 3, 4                     # encoding: [0x04,0x00,0xc3,0x08]
            tdlngi 3, 4
# CHECK-BE: td 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x88]
# CHECK-LE: td 6, 3, 4                      # encoding: [0x88,0x20,0xc3,0x7c]
            tdlng 3, 4

# CHECK-BE: twi 31, 3, 4                    # encoding: [0x0f,0xe3,0x00,0x04]
# CHECK-LE: twi 31, 3, 4                    # encoding: [0x04,0x00,0xe3,0x0f]
            twui 3, 4
# CHECK-BE: tw 31, 3, 4                     # encoding: [0x7f,0xe3,0x20,0x08]
# CHECK-LE: tw 31, 3, 4                     # encoding: [0x08,0x20,0xe3,0x7f]
            twu 3, 4
# CHECK-BE: tdi 31, 3, 4                    # encoding: [0x0b,0xe3,0x00,0x04]
# CHECK-LE: tdi 31, 3, 4                    # encoding: [0x04,0x00,0xe3,0x0b]
            tdui 3, 4
# CHECK-BE: td 31, 3, 4                     # encoding: [0x7f,0xe3,0x20,0x88]
# CHECK-LE: td 31, 3, 4                     # encoding: [0x88,0x20,0xe3,0x7f]
            tdu 3, 4

# CHECK-BE: trap                            # encoding: [0x7f,0xe0,0x00,0x08]
# CHECK-LE: trap                            # encoding: [0x08,0x00,0xe0,0x7f]
            trap

# Rotate and shift mnemonics

# CHECK-BE: rldicr 2, 3, 5, 3               # encoding: [0x78,0x62,0x28,0xc4]
# CHECK-LE: rldicr 2, 3, 5, 3               # encoding: [0xc4,0x28,0x62,0x78]
            extldi 2, 3, 4, 5
# CHECK-BE: rldicr. 2, 3, 5, 3              # encoding: [0x78,0x62,0x28,0xc5]
# CHECK-LE: rldicr. 2, 3, 5, 3              # encoding: [0xc5,0x28,0x62,0x78]
            extldi. 2, 3, 4, 5
# CHECK-BE: rldicl 2, 3, 9, 60              # encoding: [0x78,0x62,0x4f,0x20]
# CHECK-LE: rldicl 2, 3, 9, 60              # encoding: [0x20,0x4f,0x62,0x78]
            extrdi 2, 3, 4, 5
# CHECK-BE: rldicl. 2, 3, 9, 60             # encoding: [0x78,0x62,0x4f,0x21]
# CHECK-LE: rldicl. 2, 3, 9, 60             # encoding: [0x21,0x4f,0x62,0x78]
            extrdi. 2, 3, 4, 5
# CHECK-BE: rldimi 2, 3, 55, 5              # encoding: [0x78,0x62,0xb9,0x4e]
# CHECK-LE: rldimi 2, 3, 55, 5              # encoding: [0x4e,0xb9,0x62,0x78]
            insrdi 2, 3, 4, 5
# CHECK-BE: rldimi. 2, 3, 55, 5             # encoding: [0x78,0x62,0xb9,0x4f]
# CHECK-LE: rldimi. 2, 3, 55, 5             # encoding: [0x4f,0xb9,0x62,0x78]
            insrdi. 2, 3, 4, 5
# CHECK-BE: rldicl 2, 3, 4, 0               # encoding: [0x78,0x62,0x20,0x00]
# CHECK-LE: rldicl 2, 3, 4, 0               # encoding: [0x00,0x20,0x62,0x78]
            rotldi 2, 3, 4
# CHECK-BE: rldicl. 2, 3, 4, 0              # encoding: [0x78,0x62,0x20,0x01]
# CHECK-LE: rldicl. 2, 3, 4, 0              # encoding: [0x01,0x20,0x62,0x78]
            rotldi. 2, 3, 4
# CHECK-BE: rldicl 2, 3, 60, 0              # encoding: [0x78,0x62,0xe0,0x02]
# CHECK-LE: rldicl 2, 3, 60, 0              # encoding: [0x02,0xe0,0x62,0x78]
            rotrdi 2, 3, 4
# CHECK-BE: rldicl. 2, 3, 60, 0             # encoding: [0x78,0x62,0xe0,0x03]
# CHECK-LE: rldicl. 2, 3, 60, 0             # encoding: [0x03,0xe0,0x62,0x78]
            rotrdi. 2, 3, 4
# CHECK-BE: rldcl 2, 3, 4, 0                # encoding: [0x78,0x62,0x20,0x10]
# CHECK-LE: rldcl 2, 3, 4, 0                # encoding: [0x10,0x20,0x62,0x78]
            rotld 2, 3, 4
# CHECK-BE: rldcl. 2, 3, 4, 0               # encoding: [0x78,0x62,0x20,0x11]
# CHECK-LE: rldcl. 2, 3, 4, 0               # encoding: [0x11,0x20,0x62,0x78]
            rotld. 2, 3, 4
# CHECK-BE: sldi 2, 3, 4                    # encoding: [0x78,0x62,0x26,0xe4]
# CHECK-LE: sldi 2, 3, 4                    # encoding: [0xe4,0x26,0x62,0x78]
            sldi 2, 3, 4
# CHECK-BE: rldicr. 2, 3, 4, 59             # encoding: [0x78,0x62,0x26,0xe5]
# CHECK-LE: rldicr. 2, 3, 4, 59             # encoding: [0xe5,0x26,0x62,0x78]
            sldi. 2, 3, 4
# CHECK-BE: rldicl 2, 3, 60, 4              # encoding: [0x78,0x62,0xe1,0x02]
# CHECK-LE: rldicl 2, 3, 60, 4              # encoding: [0x02,0xe1,0x62,0x78]
            srdi 2, 3, 4
# CHECK-BE: rldicl. 2, 3, 60, 4             # encoding: [0x78,0x62,0xe1,0x03]
# CHECK-LE: rldicl. 2, 3, 60, 4             # encoding: [0x03,0xe1,0x62,0x78]
            srdi. 2, 3, 4
# CHECK-BE: rldicl 2, 3, 0, 4               # encoding: [0x78,0x62,0x01,0x00]
# CHECK-LE: rldicl 2, 3, 0, 4               # encoding: [0x00,0x01,0x62,0x78]
            clrldi 2, 3, 4
# CHECK-BE: rldicl. 2, 3, 0, 4              # encoding: [0x78,0x62,0x01,0x01]
# CHECK-LE: rldicl. 2, 3, 0, 4              # encoding: [0x01,0x01,0x62,0x78]
            clrldi. 2, 3, 4
# CHECK-BE: rldicr 2, 3, 0, 59              # encoding: [0x78,0x62,0x06,0xe4]
# CHECK-LE: rldicr 2, 3, 0, 59              # encoding: [0xe4,0x06,0x62,0x78]
            clrrdi 2, 3, 4
# CHECK-BE: rldicr. 2, 3, 0, 59             # encoding: [0x78,0x62,0x06,0xe5]
# CHECK-LE: rldicr. 2, 3, 0, 59             # encoding: [0xe5,0x06,0x62,0x78]
            clrrdi. 2, 3, 4
# CHECK-BE: rldic 2, 3, 4, 1                # encoding: [0x78,0x62,0x20,0x48]
# CHECK-LE: rldic 2, 3, 4, 1                # encoding: [0x48,0x20,0x62,0x78]
            clrlsldi 2, 3, 5, 4
# CHECK-BE: rldic. 2, 3, 4, 1               # encoding: [0x78,0x62,0x20,0x49]
# CHECK-LE: rldic. 2, 3, 4, 1               # encoding: [0x49,0x20,0x62,0x78]
            clrlsldi. 2, 3, 5, 4

# CHECK-BE: rlwinm 2, 3, 5, 0, 3            # encoding: [0x54,0x62,0x28,0x06]
# CHECK-LE: rlwinm 2, 3, 5, 0, 3            # encoding: [0x06,0x28,0x62,0x54]
            extlwi 2, 3, 4, 5
# CHECK-BE: rlwinm. 2, 3, 5, 0, 3           # encoding: [0x54,0x62,0x28,0x07]
# CHECK-LE: rlwinm. 2, 3, 5, 0, 3           # encoding: [0x07,0x28,0x62,0x54]
            extlwi. 2, 3, 4, 5
# CHECK-BE: rlwinm 2, 3, 9, 28, 31          # encoding: [0x54,0x62,0x4f,0x3e]
# CHECK-LE: rlwinm 2, 3, 9, 28, 31          # encoding: [0x3e,0x4f,0x62,0x54]
            extrwi 2, 3, 4, 5
# CHECK-BE: rlwinm. 2, 3, 9, 28, 31         # encoding: [0x54,0x62,0x4f,0x3f]
# CHECK-LE: rlwinm. 2, 3, 9, 28, 31         # encoding: [0x3f,0x4f,0x62,0x54]
            extrwi. 2, 3, 4, 5
# CHECK-BE: rlwimi 2, 3, 27, 5, 8           # encoding: [0x50,0x62,0xd9,0x50]
# CHECK-LE: rlwimi 2, 3, 27, 5, 8           # encoding: [0x50,0xd9,0x62,0x50]
            inslwi 2, 3, 4, 5
# CHECK-BE: rlwimi. 2, 3, 27, 5, 8          # encoding: [0x50,0x62,0xd9,0x51]
# CHECK-LE: rlwimi. 2, 3, 27, 5, 8          # encoding: [0x51,0xd9,0x62,0x50]
            inslwi. 2, 3, 4, 5
# CHECK-BE: rlwimi 2, 3, 23, 5, 8           # encoding: [0x50,0x62,0xb9,0x50]
# CHECK-LE: rlwimi 2, 3, 23, 5, 8           # encoding: [0x50,0xb9,0x62,0x50]
            insrwi 2, 3, 4, 5
# CHECK-BE: rlwimi. 2, 3, 23, 5, 8          # encoding: [0x50,0x62,0xb9,0x51]
# CHECK-LE: rlwimi. 2, 3, 23, 5, 8          # encoding: [0x51,0xb9,0x62,0x50]
            insrwi. 2, 3, 4, 5
# CHECK-BE: rlwinm 2, 3, 4, 0, 31           # encoding: [0x54,0x62,0x20,0x3e]
# CHECK-LE: rlwinm 2, 3, 4, 0, 31           # encoding: [0x3e,0x20,0x62,0x54]
            rotlwi 2, 3, 4
# CHECK-BE: rlwinm. 2, 3, 4, 0, 31          # encoding: [0x54,0x62,0x20,0x3f]
# CHECK-LE: rlwinm. 2, 3, 4, 0, 31          # encoding: [0x3f,0x20,0x62,0x54]
            rotlwi. 2, 3, 4
# CHECK-BE: rlwinm 2, 3, 28, 0, 31          # encoding: [0x54,0x62,0xe0,0x3e]
# CHECK-LE: rlwinm 2, 3, 28, 0, 31          # encoding: [0x3e,0xe0,0x62,0x54]
            rotrwi 2, 3, 4
# CHECK-BE: rlwinm. 2, 3, 28, 0, 31         # encoding: [0x54,0x62,0xe0,0x3f]
# CHECK-LE: rlwinm. 2, 3, 28, 0, 31         # encoding: [0x3f,0xe0,0x62,0x54]
            rotrwi. 2, 3, 4
# CHECK-BE: rlwnm 2, 3, 4, 0, 31            # encoding: [0x5c,0x62,0x20,0x3e]
# CHECK-LE: rlwnm 2, 3, 4, 0, 31            # encoding: [0x3e,0x20,0x62,0x5c]
            rotlw 2, 3, 4
# CHECK-BE: rlwnm. 2, 3, 4, 0, 31           # encoding: [0x5c,0x62,0x20,0x3f]
# CHECK-LE: rlwnm. 2, 3, 4, 0, 31           # encoding: [0x3f,0x20,0x62,0x5c]
            rotlw. 2, 3, 4
# CHECK-BE: slwi 2, 3, 4                    # encoding: [0x54,0x62,0x20,0x36]
# CHECK-LE: slwi 2, 3, 4                    # encoding: [0x36,0x20,0x62,0x54]
            slwi 2, 3, 4
# CHECK-BE: rlwinm. 2, 3, 4, 0, 27          # encoding: [0x54,0x62,0x20,0x37]
# CHECK-LE: rlwinm. 2, 3, 4, 0, 27          # encoding: [0x37,0x20,0x62,0x54]
            slwi. 2, 3, 4
# CHECK-BE: srwi 2, 3, 4                    # encoding: [0x54,0x62,0xe1,0x3e]
# CHECK-LE: srwi 2, 3, 4                    # encoding: [0x3e,0xe1,0x62,0x54]
            srwi 2, 3, 4
# CHECK-BE: rlwinm. 2, 3, 28, 4, 31         # encoding: [0x54,0x62,0xe1,0x3f]
# CHECK-LE: rlwinm. 2, 3, 28, 4, 31         # encoding: [0x3f,0xe1,0x62,0x54]
            srwi. 2, 3, 4
# CHECK-BE: rlwinm 2, 3, 0, 4, 31           # encoding: [0x54,0x62,0x01,0x3e]
# CHECK-LE: rlwinm 2, 3, 0, 4, 31           # encoding: [0x3e,0x01,0x62,0x54]
            clrlwi 2, 3, 4
# CHECK-BE: rlwinm. 2, 3, 0, 4, 31          # encoding: [0x54,0x62,0x01,0x3f]
# CHECK-LE: rlwinm. 2, 3, 0, 4, 31          # encoding: [0x3f,0x01,0x62,0x54]
            clrlwi. 2, 3, 4
# CHECK-BE: rlwinm 2, 3, 0, 0, 27           # encoding: [0x54,0x62,0x00,0x36]
# CHECK-LE: rlwinm 2, 3, 0, 0, 27           # encoding: [0x36,0x00,0x62,0x54]
            clrrwi 2, 3, 4
# CHECK-BE: rlwinm. 2, 3, 0, 0, 27          # encoding: [0x54,0x62,0x00,0x37]
# CHECK-LE: rlwinm. 2, 3, 0, 0, 27          # encoding: [0x37,0x00,0x62,0x54]
            clrrwi. 2, 3, 4
# CHECK-BE: rlwinm 2, 3, 4, 1, 27           # encoding: [0x54,0x62,0x20,0x76]
# CHECK-LE: rlwinm 2, 3, 4, 1, 27           # encoding: [0x76,0x20,0x62,0x54]
            clrlslwi 2, 3, 5, 4
# CHECK-BE: rlwinm. 2, 3, 4, 1, 27          # encoding: [0x54,0x62,0x20,0x77]
# CHECK-LE: rlwinm. 2, 3, 4, 1, 27          # encoding: [0x77,0x20,0x62,0x54]
            clrlslwi. 2, 3, 5, 4

# Move to/from special purpose register mnemonics

# CHECK-BE: mtspr 1, 2                      # encoding: [0x7c,0x41,0x03,0xa6]
# CHECK-LE: mtspr 1, 2                      # encoding: [0xa6,0x03,0x41,0x7c]
            mtxer 2
# CHECK-BE: mfspr 2, 1                      # encoding: [0x7c,0x41,0x02,0xa6]
# CHECK-LE: mfspr 2, 1                      # encoding: [0xa6,0x02,0x41,0x7c]
            mfxer 2
# CHECK-BE: mtlr 2                          # encoding: [0x7c,0x48,0x03,0xa6]
# CHECK-LE: mtlr 2                          # encoding: [0xa6,0x03,0x48,0x7c]
            mtlr 2
# CHECK-BE: mflr 2                          # encoding: [0x7c,0x48,0x02,0xa6]
# CHECK-LE: mflr 2                          # encoding: [0xa6,0x02,0x48,0x7c]
            mflr 2
# CHECK-BE: mtctr 2                         # encoding: [0x7c,0x49,0x03,0xa6]
# CHECK-LE: mtctr 2                         # encoding: [0xa6,0x03,0x49,0x7c]
            mtctr 2
# CHECK-BE: mfctr 2                         # encoding: [0x7c,0x49,0x02,0xa6]
# CHECK-LE: mfctr 2                         # encoding: [0xa6,0x02,0x49,0x7c]
            mfctr 2

# Miscellaneous mnemonics

# CHECK-BE: nop                             # encoding: [0x60,0x00,0x00,0x00]
# CHECK-LE: nop                             # encoding: [0x00,0x00,0x00,0x60]
            nop
# CHECK-BE: xori 0, 0, 0                    # encoding: [0x68,0x00,0x00,0x00]
# CHECK-LE: xori 0, 0, 0                    # encoding: [0x00,0x00,0x00,0x68]
            xnop
# CHECK-BE: li 2, 128                       # encoding: [0x38,0x40,0x00,0x80]
# CHECK-LE: li 2, 128                       # encoding: [0x80,0x00,0x40,0x38]
            li 2, 128
# CHECK-BE: lis 2, 128                      # encoding: [0x3c,0x40,0x00,0x80]
# CHECK-LE: lis 2, 128                      # encoding: [0x80,0x00,0x40,0x3c]
            lis 2, 128
# CHECK-BE: la 2, 128(4)
# CHECK-LE: la 2, 128(4)
            la 2, 128(4)
# CHECK-BE: mr 2, 3                         # encoding: [0x7c,0x62,0x1b,0x78]
# CHECK-LE: mr 2, 3                         # encoding: [0x78,0x1b,0x62,0x7c]
            mr 2, 3
# CHECK-BE: or. 2, 3, 3                     # encoding: [0x7c,0x62,0x1b,0x79]
# CHECK-LE: or. 2, 3, 3                     # encoding: [0x79,0x1b,0x62,0x7c]
            mr. 2, 3
# CHECK-BE: nor 2, 3, 3                     # encoding: [0x7c,0x62,0x18,0xf8]
# CHECK-LE: nor 2, 3, 3                     # encoding: [0xf8,0x18,0x62,0x7c]
            not 2, 3
# CHECK-BE: nor. 2, 3, 3                    # encoding: [0x7c,0x62,0x18,0xf9]
# CHECK-LE: nor. 2, 3, 3                    # encoding: [0xf9,0x18,0x62,0x7c]
            not. 2, 3
# CHECK-BE: mtcrf 255, 2                    # encoding: [0x7c,0x4f,0xf1,0x20]
# CHECK-LE: mtcrf 255, 2                    # encoding: [0x20,0xf1,0x4f,0x7c]
            mtcr 2

