
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# Condition register bit symbols

# CHECK: beqlr 0                         # encoding: [0x4d,0x82,0x00,0x20]
         beqlr cr0
# CHECK: beqlr 1                         # encoding: [0x4d,0x86,0x00,0x20]
         beqlr cr1
# CHECK: beqlr 2                         # encoding: [0x4d,0x8a,0x00,0x20]
         beqlr cr2
# CHECK: beqlr 3                         # encoding: [0x4d,0x8e,0x00,0x20]
         beqlr cr3
# CHECK: beqlr 4                         # encoding: [0x4d,0x92,0x00,0x20]
         beqlr cr4
# CHECK: beqlr 5                         # encoding: [0x4d,0x96,0x00,0x20]
         beqlr cr5
# CHECK: beqlr 6                         # encoding: [0x4d,0x9a,0x00,0x20]
         beqlr cr6
# CHECK: beqlr 7                         # encoding: [0x4d,0x9e,0x00,0x20]
         beqlr cr7

# CHECK: bclr 12, 0, 0                   # encoding: [0x4d,0x80,0x00,0x20]
         btlr 4*cr0+lt
# CHECK: bclr 12, 1, 0                   # encoding: [0x4d,0x81,0x00,0x20]
         btlr 4*cr0+gt
# CHECK: bclr 12, 2, 0                   # encoding: [0x4d,0x82,0x00,0x20]
         btlr 4*cr0+eq
# CHECK: bclr 12, 3, 0                   # encoding: [0x4d,0x83,0x00,0x20]
         btlr 4*cr0+so
# CHECK: bclr 12, 3, 0                   # encoding: [0x4d,0x83,0x00,0x20]
         btlr 4*cr0+un
# CHECK: bclr 12, 4, 0                   # encoding: [0x4d,0x84,0x00,0x20]
         btlr 4*cr1+lt
# CHECK: bclr 12, 5, 0                   # encoding: [0x4d,0x85,0x00,0x20]
         btlr 4*cr1+gt
# CHECK: bclr 12, 6, 0                   # encoding: [0x4d,0x86,0x00,0x20]
         btlr 4*cr1+eq
# CHECK: bclr 12, 7, 0                   # encoding: [0x4d,0x87,0x00,0x20]
         btlr 4*cr1+so
# CHECK: bclr 12, 7, 0                   # encoding: [0x4d,0x87,0x00,0x20]
         btlr 4*cr1+un
# CHECK: bclr 12, 8, 0                   # encoding: [0x4d,0x88,0x00,0x20]
         btlr 4*cr2+lt
# CHECK: bclr 12, 9, 0                   # encoding: [0x4d,0x89,0x00,0x20]
         btlr 4*cr2+gt
# CHECK: bclr 12, 10, 0                  # encoding: [0x4d,0x8a,0x00,0x20]
         btlr 4*cr2+eq
# CHECK: bclr 12, 11, 0                  # encoding: [0x4d,0x8b,0x00,0x20]
         btlr 4*cr2+so
# CHECK: bclr 12, 11, 0                  # encoding: [0x4d,0x8b,0x00,0x20]
         btlr 4*cr2+un
# CHECK: bclr 12, 12, 0                  # encoding: [0x4d,0x8c,0x00,0x20]
         btlr 4*cr3+lt
# CHECK: bclr 12, 13, 0                  # encoding: [0x4d,0x8d,0x00,0x20]
         btlr 4*cr3+gt
# CHECK: bclr 12, 14, 0                  # encoding: [0x4d,0x8e,0x00,0x20]
         btlr 4*cr3+eq
# CHECK: bclr 12, 15, 0                  # encoding: [0x4d,0x8f,0x00,0x20]
         btlr 4*cr3+so
# CHECK: bclr 12, 15, 0                  # encoding: [0x4d,0x8f,0x00,0x20]
         btlr 4*cr3+un
# CHECK: bclr 12, 16, 0                  # encoding: [0x4d,0x90,0x00,0x20]
         btlr 4*cr4+lt
# CHECK: bclr 12, 17, 0                  # encoding: [0x4d,0x91,0x00,0x20]
         btlr 4*cr4+gt
# CHECK: bclr 12, 18, 0                  # encoding: [0x4d,0x92,0x00,0x20]
         btlr 4*cr4+eq
# CHECK: bclr 12, 19, 0                  # encoding: [0x4d,0x93,0x00,0x20]
         btlr 4*cr4+so
# CHECK: bclr 12, 19, 0                  # encoding: [0x4d,0x93,0x00,0x20]
         btlr 4*cr4+un
# CHECK: bclr 12, 20, 0                  # encoding: [0x4d,0x94,0x00,0x20]
         btlr 4*cr5+lt
# CHECK: bclr 12, 21, 0                  # encoding: [0x4d,0x95,0x00,0x20]
         btlr 4*cr5+gt
# CHECK: bclr 12, 22, 0                  # encoding: [0x4d,0x96,0x00,0x20]
         btlr 4*cr5+eq
# CHECK: bclr 12, 23, 0                  # encoding: [0x4d,0x97,0x00,0x20]
         btlr 4*cr5+so
# CHECK: bclr 12, 23, 0                  # encoding: [0x4d,0x97,0x00,0x20]
         btlr 4*cr5+un
# CHECK: bclr 12, 24, 0                  # encoding: [0x4d,0x98,0x00,0x20]
         btlr 4*cr6+lt
# CHECK: bclr 12, 25, 0                  # encoding: [0x4d,0x99,0x00,0x20]
         btlr 4*cr6+gt
# CHECK: bclr 12, 26, 0                  # encoding: [0x4d,0x9a,0x00,0x20]
         btlr 4*cr6+eq
# CHECK: bclr 12, 27, 0                  # encoding: [0x4d,0x9b,0x00,0x20]
         btlr 4*cr6+so
# CHECK: bclr 12, 27, 0                  # encoding: [0x4d,0x9b,0x00,0x20]
         btlr 4*cr6+un
# CHECK: bclr 12, 28, 0                  # encoding: [0x4d,0x9c,0x00,0x20]
         btlr 4*cr7+lt
# CHECK: bclr 12, 29, 0                  # encoding: [0x4d,0x9d,0x00,0x20]
         btlr 4*cr7+gt
# CHECK: bclr 12, 30, 0                  # encoding: [0x4d,0x9e,0x00,0x20]
         btlr 4*cr7+eq
# CHECK: bclr 12, 31, 0                  # encoding: [0x4d,0x9f,0x00,0x20]
         btlr 4*cr7+so
# CHECK: bclr 12, 31, 0                  # encoding: [0x4d,0x9f,0x00,0x20]
         btlr 4*cr7+un

# Branch mnemonics

# CHECK: blr                             # encoding: [0x4e,0x80,0x00,0x20]
         blr
# CHECK: bctr                            # encoding: [0x4e,0x80,0x04,0x20]
         bctr
# CHECK: blrl                            # encoding: [0x4e,0x80,0x00,0x21]
         blrl
# CHECK: bctrl                           # encoding: [0x4e,0x80,0x04,0x21]
         bctrl

# CHECK: bc 12, 2, target                # encoding: [0x41,0x82,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bt 2, target
# CHECK: bca 12, 2, target               # encoding: [0x41,0x82,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bta 2, target
# CHECK: bclr 12, 2, 0                   # encoding: [0x4d,0x82,0x00,0x20]
         btlr 2
# CHECK: bcctr 12, 2, 0                  # encoding: [0x4d,0x82,0x04,0x20]
         btctr 2
# CHECK: bcl 12, 2, target               # encoding: [0x41,0x82,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         btl 2, target
# CHECK: bcla 12, 2, target              # encoding: [0x41,0x82,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         btla 2, target
# CHECK: bclrl 12, 2, 0                  # encoding: [0x4d,0x82,0x00,0x21]
         btlrl 2
# CHECK: bcctrl 12, 2, 0                 # encoding: [0x4d,0x82,0x04,0x21]
         btctrl 2

# CHECK: bc 15, 2, target                # encoding: [0x41,0xe2,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bt+ 2, target
# CHECK: bca 15, 2, target               # encoding: [0x41,0xe2,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bta+ 2, target
# CHECK: bclr 15, 2, 0                   # encoding: [0x4d,0xe2,0x00,0x20]
         btlr+ 2
# CHECK: bcctr 15, 2, 0                  # encoding: [0x4d,0xe2,0x04,0x20]
         btctr+ 2
# CHECK: bcl 15, 2, target               # encoding: [0x41,0xe2,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         btl+ 2, target
# CHECK: bcla 15, 2, target              # encoding: [0x41,0xe2,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         btla+ 2, target
# CHECK: bclrl 15, 2, 0                  # encoding: [0x4d,0xe2,0x00,0x21]
         btlrl+ 2
# CHECK: bcctrl 15, 2, 0                 # encoding: [0x4d,0xe2,0x04,0x21]
         btctrl+ 2

# CHECK: bc 14, 2, target                # encoding: [0x41,0xc2,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bt- 2, target
# CHECK: bca 14, 2, target               # encoding: [0x41,0xc2,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bta- 2, target
# CHECK: bclr 14, 2, 0                   # encoding: [0x4d,0xc2,0x00,0x20]
         btlr- 2
# CHECK: bcctr 14, 2, 0                  # encoding: [0x4d,0xc2,0x04,0x20]
         btctr- 2
# CHECK: bcl 14, 2, target               # encoding: [0x41,0xc2,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         btl- 2, target
# CHECK: bcla 14, 2, target              # encoding: [0x41,0xc2,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         btla- 2, target
# CHECK: bclrl 14, 2, 0                  # encoding: [0x4d,0xc2,0x00,0x21]
         btlrl- 2
# CHECK: bcctrl 14, 2, 0                 # encoding: [0x4d,0xc2,0x04,0x21]
         btctrl- 2

# CHECK: bc 4, 2, target                 # encoding: [0x40,0x82,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bf 2, target
# CHECK: bca 4, 2, target                # encoding: [0x40,0x82,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bfa 2, target
# CHECK: bclr 4, 2, 0                    # encoding: [0x4c,0x82,0x00,0x20]
         bflr 2
# CHECK: bcctr 4, 2, 0                   # encoding: [0x4c,0x82,0x04,0x20]
         bfctr 2
# CHECK: bcl 4, 2, target                # encoding: [0x40,0x82,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bfl 2, target
# CHECK: bcla 4, 2, target               # encoding: [0x40,0x82,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bfla 2, target
# CHECK: bclrl 4, 2, 0                   # encoding: [0x4c,0x82,0x00,0x21]
         bflrl 2
# CHECK: bcctrl 4, 2, 0                  # encoding: [0x4c,0x82,0x04,0x21]
         bfctrl 2

# CHECK: bc 7, 2, target                 # encoding: [0x40,0xe2,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bf+ 2, target
# CHECK: bca 7, 2, target                # encoding: [0x40,0xe2,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bfa+ 2, target
# CHECK: bclr 7, 2, 0                    # encoding: [0x4c,0xe2,0x00,0x20]
         bflr+ 2
# CHECK: bcctr 7, 2, 0                   # encoding: [0x4c,0xe2,0x04,0x20]
         bfctr+ 2
# CHECK: bcl 7, 2, target                # encoding: [0x40,0xe2,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bfl+ 2, target
# CHECK: bcla 7, 2, target               # encoding: [0x40,0xe2,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bfla+ 2, target
# CHECK: bclrl 7, 2, 0                   # encoding: [0x4c,0xe2,0x00,0x21]
         bflrl+ 2
# CHECK: bcctrl 7, 2, 0                  # encoding: [0x4c,0xe2,0x04,0x21]
         bfctrl+ 2

# CHECK: bc 6, 2, target                 # encoding: [0x40,0xc2,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bf- 2, target
# CHECK: bca 6, 2, target                # encoding: [0x40,0xc2,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bfa- 2, target
# CHECK: bclr 6, 2, 0                    # encoding: [0x4c,0xc2,0x00,0x20]
         bflr- 2
# CHECK: bcctr 6, 2, 0                   # encoding: [0x4c,0xc2,0x04,0x20]
         bfctr- 2
# CHECK: bcl 6, 2, target                # encoding: [0x40,0xc2,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bfl- 2, target
# CHECK: bcla 6, 2, target               # encoding: [0x40,0xc2,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bfla- 2, target
# CHECK: bclrl 6, 2, 0                   # encoding: [0x4c,0xc2,0x00,0x21]
         bflrl- 2
# CHECK: bcctrl 6, 2, 0                  # encoding: [0x4c,0xc2,0x04,0x21]
         bfctrl- 2

# CHECK: bdnz target                     # encoding: [0x42,0x00,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnz target
# CHECK: bdnza target                    # encoding: [0x42,0x00,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdnza target
# CHECK: bdnzlr                          # encoding: [0x4e,0x00,0x00,0x20]
         bdnzlr
# CHECK: bdnzl target                    # encoding: [0x42,0x00,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnzl target
# CHECK: bdnzla target                   # encoding: [0x42,0x00,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdnzla target
# CHECK: bdnzlrl                         # encoding: [0x4e,0x00,0x00,0x21]
         bdnzlrl

# CHECK: bdnz+ target                    # encoding: [0x43,0x20,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnz+ target
# CHECK: bdnza+ target                   # encoding: [0x43,0x20,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdnza+ target
# CHECK: bdnzlr+                         # encoding: [0x4f,0x20,0x00,0x20]
         bdnzlr+
# CHECK: bdnzl+ target                   # encoding: [0x43,0x20,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnzl+ target
# CHECK: bdnzla+ target                  # encoding: [0x43,0x20,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdnzla+ target
# CHECK: bdnzlrl+                        # encoding: [0x4f,0x20,0x00,0x21]
         bdnzlrl+

# CHECK: bdnz- target                    # encoding: [0x43,0x00,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnz- target
# CHECK: bdnza- target                   # encoding: [0x43,0x00,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdnza- target
# CHECK: bdnzlr-                         # encoding: [0x4f,0x00,0x00,0x20]
         bdnzlr-
# CHECK: bdnzl- target                   # encoding: [0x43,0x00,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnzl- target
# CHECK: bdnzla- target                  # encoding: [0x43,0x00,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdnzla- target
# CHECK: bdnzlrl-                        # encoding: [0x4f,0x00,0x00,0x21]
         bdnzlrl-

# CHECK: bc 8, 2, target                 # encoding: [0x41,0x02,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnzt 2, target
# CHECK: bca 8, 2, target                # encoding: [0x41,0x02,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdnzta 2, target
# CHECK: bclr 8, 2, 0                    # encoding: [0x4d,0x02,0x00,0x20]
         bdnztlr 2
# CHECK: bcl 8, 2, target                # encoding: [0x41,0x02,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnztl 2, target
# CHECK: bcla 8, 2, target               # encoding: [0x41,0x02,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdnztla 2, target
# CHECK: bclrl 8, 2, 0                   # encoding: [0x4d,0x02,0x00,0x21]
         bdnztlrl 2

# CHECK: bc 0, 2, target                 # encoding: [0x40,0x02,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnzf 2, target
# CHECK: bca 0, 2, target                # encoding: [0x40,0x02,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdnzfa 2, target
# CHECK: bclr 0, 2, 0                    # encoding: [0x4c,0x02,0x00,0x20]
         bdnzflr 2
# CHECK: bcl 0, 2, target                # encoding: [0x40,0x02,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnzfl 2, target
# CHECK: bcla 0, 2, target               # encoding: [0x40,0x02,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdnzfla 2, target
# CHECK: bclrl 0, 2, 0                   # encoding: [0x4c,0x02,0x00,0x21]
         bdnzflrl 2

# CHECK: bdz target                      # encoding: [0x42,0x40,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdz target
# CHECK: bdza target                     # encoding: [0x42,0x40,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdza target
# CHECK: bdzlr                           # encoding: [0x4e,0x40,0x00,0x20]
         bdzlr
# CHECK: bdzl target                     # encoding: [0x42,0x40,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdzl target
# CHECK: bdzla target                    # encoding: [0x42,0x40,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdzla target
# CHECK: bdzlrl                          # encoding: [0x4e,0x40,0x00,0x21]
         bdzlrl

# CHECK: bdz+ target                     # encoding: [0x43,0x60,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdz+ target
# CHECK: bdza+ target                    # encoding: [0x43,0x60,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdza+ target
# CHECK: bdzlr+                          # encoding: [0x4f,0x60,0x00,0x20]
         bdzlr+
# CHECK: bdzl+ target                    # encoding: [0x43,0x60,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdzl+ target
# CHECK: bdzla+ target                   # encoding: [0x43,0x60,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdzla+ target
# CHECK: bdzlrl+                         # encoding: [0x4f,0x60,0x00,0x21]
         bdzlrl+

# CHECK: bdz- target                     # encoding: [0x43,0x40,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdz- target
# CHECK: bdza- target                    # encoding: [0x43,0x40,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdza- target
# CHECK: bdzlr-                          # encoding: [0x4f,0x40,0x00,0x20]
         bdzlr-
# CHECK: bdzl- target                    # encoding: [0x43,0x40,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdzl- target
# CHECK: bdzla- target                   # encoding: [0x43,0x40,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdzla- target
# CHECK: bdzlrl-                         # encoding: [0x4f,0x40,0x00,0x21]
         bdzlrl-

# CHECK: bc 10, 2, target                # encoding: [0x41,0x42,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdzt 2, target
# CHECK: bca 10, 2, target               # encoding: [0x41,0x42,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdzta 2, target
# CHECK: bclr 10, 2, 0                   # encoding: [0x4d,0x42,0x00,0x20]
         bdztlr 2
# CHECK: bcl 10, 2, target               # encoding: [0x41,0x42,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdztl 2, target
# CHECK: bcla 10, 2, target              # encoding: [0x41,0x42,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdztla 2, target
# CHECK: bclrl 10, 2, 0                  # encoding: [0x4d,0x42,0x00,0x21]
         bdztlrl 2

# CHECK: bc 2, 2, target                 # encoding: [0x40,0x42,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdzf 2, target
# CHECK: bca 2, 2, target                # encoding: [0x40,0x42,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdzfa 2, target
# CHECK: bclr 2, 2, 0                    # encoding: [0x4c,0x42,0x00,0x20]
         bdzflr 2
# CHECK: bcl 2, 2, target                # encoding: [0x40,0x42,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdzfl 2, target
# CHECK: bcla 2, 2, target               # encoding: [0x40,0x42,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bdzfla 2, target
# CHECK: bclrl 2, 2, 0                   # encoding: [0x4c,0x42,0x00,0x21]
         bdzflrl 2

# CHECK: blt 2, target                   # encoding: [0x41,0x88,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blt 2, target
# CHECK: blt 0, target                   # encoding: [0x41,0x80,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blt target
# CHECK: blta 2, target                  # encoding: [0x41,0x88,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blta 2, target
# CHECK: blta 0, target                  # encoding: [0x41,0x80,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blta target
# CHECK: bltlr 2                         # encoding: [0x4d,0x88,0x00,0x20]
         bltlr 2
# CHECK: bltlr 0                         # encoding: [0x4d,0x80,0x00,0x20]
         bltlr
# CHECK: bltctr 2                        # encoding: [0x4d,0x88,0x04,0x20]
         bltctr 2
# CHECK: bltctr 0                        # encoding: [0x4d,0x80,0x04,0x20]
         bltctr
# CHECK: bltl 2, target                  # encoding: [0x41,0x88,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bltl 2, target
# CHECK: bltl 0, target                  # encoding: [0x41,0x80,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bltl target
# CHECK: bltla 2, target                 # encoding: [0x41,0x88,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bltla 2, target
# CHECK: bltla 0, target                 # encoding: [0x41,0x80,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bltla target
# CHECK: bltlrl 2                        # encoding: [0x4d,0x88,0x00,0x21]
         bltlrl 2
# CHECK: bltlrl 0                        # encoding: [0x4d,0x80,0x00,0x21]
         bltlrl
# CHECK: bltctrl 2                       # encoding: [0x4d,0x88,0x04,0x21]
         bltctrl 2
# CHECK: bltctrl 0                       # encoding: [0x4d,0x80,0x04,0x21]
         bltctrl

# CHECK: blt+ 2, target                  # encoding: [0x41,0xe8,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blt+ 2, target
# CHECK: blt+ 0, target                  # encoding: [0x41,0xe0,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blt+ target
# CHECK: blta+ 2, target                 # encoding: [0x41,0xe8,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blta+ 2, target
# CHECK: blta+ 0, target                 # encoding: [0x41,0xe0,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blta+ target
# CHECK: bltlr+ 2                        # encoding: [0x4d,0xe8,0x00,0x20]
         bltlr+ 2
# CHECK: bltlr+ 0                        # encoding: [0x4d,0xe0,0x00,0x20]
         bltlr+
# CHECK: bltctr+ 2                       # encoding: [0x4d,0xe8,0x04,0x20]
         bltctr+ 2
# CHECK: bltctr+ 0                       # encoding: [0x4d,0xe0,0x04,0x20]
         bltctr+
# CHECK: bltl+ 2, target                 # encoding: [0x41,0xe8,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bltl+ 2, target
# CHECK: bltl+ 0, target                 # encoding: [0x41,0xe0,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bltl+ target
# CHECK: bltla+ 2, target                # encoding: [0x41,0xe8,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bltla+ 2, target
# CHECK: bltla+ 0, target                # encoding: [0x41,0xe0,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bltla+ target
# CHECK: bltlrl+ 2                       # encoding: [0x4d,0xe8,0x00,0x21]
         bltlrl+ 2
# CHECK: bltlrl+ 0                       # encoding: [0x4d,0xe0,0x00,0x21]
         bltlrl+
# CHECK: bltctrl+ 2                      # encoding: [0x4d,0xe8,0x04,0x21]
         bltctrl+ 2
# CHECK: bltctrl+ 0                      # encoding: [0x4d,0xe0,0x04,0x21]
         bltctrl+

# CHECK: blt- 2, target                  # encoding: [0x41,0xc8,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blt- 2, target
# CHECK: blt- 0, target                  # encoding: [0x41,0xc0,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blt- target
# CHECK: blta- 2, target                 # encoding: [0x41,0xc8,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blta- 2, target
# CHECK: blta- 0, target                 # encoding: [0x41,0xc0,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blta- target
# CHECK: bltlr- 2                        # encoding: [0x4d,0xc8,0x00,0x20]
         bltlr- 2
# CHECK: bltlr- 0                        # encoding: [0x4d,0xc0,0x00,0x20]
         bltlr-
# CHECK: bltctr- 2                       # encoding: [0x4d,0xc8,0x04,0x20]
         bltctr- 2
# CHECK: bltctr- 0                       # encoding: [0x4d,0xc0,0x04,0x20]
         bltctr-
# CHECK: bltl- 2, target                 # encoding: [0x41,0xc8,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bltl- 2, target
# CHECK: bltl- 0, target                 # encoding: [0x41,0xc0,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bltl- target
# CHECK: bltla- 2, target                # encoding: [0x41,0xc8,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bltla- 2, target
# CHECK: bltla- 0, target                # encoding: [0x41,0xc0,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bltla- target
# CHECK: bltlrl- 2                       # encoding: [0x4d,0xc8,0x00,0x21]
         bltlrl- 2
# CHECK: bltlrl- 0                       # encoding: [0x4d,0xc0,0x00,0x21]
         bltlrl-
# CHECK: bltctrl- 2                      # encoding: [0x4d,0xc8,0x04,0x21]
         bltctrl- 2
# CHECK: bltctrl- 0                      # encoding: [0x4d,0xc0,0x04,0x21]
         bltctrl-

# CHECK: ble 2, target                   # encoding: [0x40,0x89,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         ble 2, target
# CHECK: ble 0, target                   # encoding: [0x40,0x81,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         ble target
# CHECK: blea 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blea 2, target
# CHECK: blea 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blea target
# CHECK: blelr 2                         # encoding: [0x4c,0x89,0x00,0x20]
         blelr 2
# CHECK: blelr 0                         # encoding: [0x4c,0x81,0x00,0x20]
         blelr
# CHECK: blectr 2                        # encoding: [0x4c,0x89,0x04,0x20]
         blectr 2
# CHECK: blectr 0                        # encoding: [0x4c,0x81,0x04,0x20]
         blectr
# CHECK: blel 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blel 2, target
# CHECK: blel 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blel target
# CHECK: blela 2, target                 # encoding: [0x40,0x89,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blela 2, target
# CHECK: blela 0, target                 # encoding: [0x40,0x81,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blela target
# CHECK: blelrl 2                        # encoding: [0x4c,0x89,0x00,0x21]
         blelrl 2
# CHECK: blelrl 0                        # encoding: [0x4c,0x81,0x00,0x21]
         blelrl
# CHECK: blectrl 2                       # encoding: [0x4c,0x89,0x04,0x21]
         blectrl 2
# CHECK: blectrl 0                       # encoding: [0x4c,0x81,0x04,0x21]
         blectrl

# CHECK: ble+ 2, target                  # encoding: [0x40,0xe9,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         ble+ 2, target
# CHECK: ble+ 0, target                  # encoding: [0x40,0xe1,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         ble+ target
# CHECK: blea+ 2, target                 # encoding: [0x40,0xe9,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blea+ 2, target
# CHECK: blea+ 0, target                 # encoding: [0x40,0xe1,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blea+ target
# CHECK: blelr+ 2                        # encoding: [0x4c,0xe9,0x00,0x20]
         blelr+ 2
# CHECK: blelr+ 0                        # encoding: [0x4c,0xe1,0x00,0x20]
         blelr+
# CHECK: blectr+ 2                       # encoding: [0x4c,0xe9,0x04,0x20]
         blectr+ 2
# CHECK: blectr+ 0                       # encoding: [0x4c,0xe1,0x04,0x20]
         blectr+
# CHECK: blel+ 2, target                 # encoding: [0x40,0xe9,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blel+ 2, target
# CHECK: blel+ 0, target                 # encoding: [0x40,0xe1,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blel+ target
# CHECK: blela+ 2, target                # encoding: [0x40,0xe9,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blela+ 2, target
# CHECK: blela+ 0, target                # encoding: [0x40,0xe1,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blela+ target
# CHECK: blelrl+ 2                       # encoding: [0x4c,0xe9,0x00,0x21]
         blelrl+ 2
# CHECK: blelrl+ 0                       # encoding: [0x4c,0xe1,0x00,0x21]
         blelrl+
# CHECK: blectrl+ 2                      # encoding: [0x4c,0xe9,0x04,0x21]
         blectrl+ 2
# CHECK: blectrl+ 0                      # encoding: [0x4c,0xe1,0x04,0x21]
         blectrl+

# CHECK: ble- 2, target                  # encoding: [0x40,0xc9,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         ble- 2, target
# CHECK: ble- 0, target                  # encoding: [0x40,0xc1,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         ble- target
# CHECK: blea- 2, target                 # encoding: [0x40,0xc9,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blea- 2, target
# CHECK: blea- 0, target                 # encoding: [0x40,0xc1,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blea- target
# CHECK: blelr- 2                        # encoding: [0x4c,0xc9,0x00,0x20]
         blelr- 2
# CHECK: blelr- 0                        # encoding: [0x4c,0xc1,0x00,0x20]
         blelr-
# CHECK: blectr- 2                       # encoding: [0x4c,0xc9,0x04,0x20]
         blectr- 2
# CHECK: blectr- 0                       # encoding: [0x4c,0xc1,0x04,0x20]
         blectr-
# CHECK: blel- 2, target                 # encoding: [0x40,0xc9,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blel- 2, target
# CHECK: blel- 0, target                 # encoding: [0x40,0xc1,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blel- target
# CHECK: blela- 2, target                # encoding: [0x40,0xc9,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blela- 2, target
# CHECK: blela- 0, target                # encoding: [0x40,0xc1,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         blela- target
# CHECK: blelrl- 2                       # encoding: [0x4c,0xc9,0x00,0x21]
         blelrl- 2
# CHECK: blelrl- 0                       # encoding: [0x4c,0xc1,0x00,0x21]
         blelrl-
# CHECK: blectrl- 2                      # encoding: [0x4c,0xc9,0x04,0x21]
         blectrl- 2
# CHECK: blectrl- 0                      # encoding: [0x4c,0xc1,0x04,0x21]
         blectrl-

# CHECK: beq 2, target                   # encoding: [0x41,0x8a,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beq 2, target
# CHECK: beq 0, target                   # encoding: [0x41,0x82,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beq target
# CHECK: beqa 2, target                  # encoding: [0x41,0x8a,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqa 2, target
# CHECK: beqa 0, target                  # encoding: [0x41,0x82,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqa target
# CHECK: beqlr 2                         # encoding: [0x4d,0x8a,0x00,0x20]
         beqlr 2
# CHECK: beqlr 0                         # encoding: [0x4d,0x82,0x00,0x20]
         beqlr
# CHECK: beqctr 2                        # encoding: [0x4d,0x8a,0x04,0x20]
         beqctr 2
# CHECK: beqctr 0                        # encoding: [0x4d,0x82,0x04,0x20]
         beqctr
# CHECK: beql 2, target                  # encoding: [0x41,0x8a,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beql 2, target
# CHECK: beql 0, target                  # encoding: [0x41,0x82,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beql target
# CHECK: beqla 2, target                 # encoding: [0x41,0x8a,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqla 2, target
# CHECK: beqla 0, target                 # encoding: [0x41,0x82,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqla target
# CHECK: beqlrl 2                        # encoding: [0x4d,0x8a,0x00,0x21]
         beqlrl 2
# CHECK: beqlrl 0                        # encoding: [0x4d,0x82,0x00,0x21]
         beqlrl
# CHECK: beqctrl 2                       # encoding: [0x4d,0x8a,0x04,0x21]
         beqctrl 2
# CHECK: beqctrl 0                       # encoding: [0x4d,0x82,0x04,0x21]
         beqctrl

# CHECK: beq+ 2, target                  # encoding: [0x41,0xea,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beq+ 2, target
# CHECK: beq+ 0, target                  # encoding: [0x41,0xe2,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beq+ target
# CHECK: beqa+ 2, target                 # encoding: [0x41,0xea,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqa+ 2, target
# CHECK: beqa+ 0, target                 # encoding: [0x41,0xe2,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqa+ target
# CHECK: beqlr+ 2                        # encoding: [0x4d,0xea,0x00,0x20]
         beqlr+ 2
# CHECK: beqlr+ 0                        # encoding: [0x4d,0xe2,0x00,0x20]
         beqlr+
# CHECK: beqctr+ 2                       # encoding: [0x4d,0xea,0x04,0x20]
         beqctr+ 2
# CHECK: beqctr+ 0                       # encoding: [0x4d,0xe2,0x04,0x20]
         beqctr+
# CHECK: beql+ 2, target                 # encoding: [0x41,0xea,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beql+ 2, target
# CHECK: beql+ 0, target                 # encoding: [0x41,0xe2,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beql+ target
# CHECK: beqla+ 2, target                # encoding: [0x41,0xea,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqla+ 2, target
# CHECK: beqla+ 0, target                # encoding: [0x41,0xe2,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqla+ target
# CHECK: beqlrl+ 2                       # encoding: [0x4d,0xea,0x00,0x21]
         beqlrl+ 2
# CHECK: beqlrl+ 0                       # encoding: [0x4d,0xe2,0x00,0x21]
         beqlrl+
# CHECK: beqctrl+ 2                      # encoding: [0x4d,0xea,0x04,0x21]
         beqctrl+ 2
# CHECK: beqctrl+ 0                      # encoding: [0x4d,0xe2,0x04,0x21]
         beqctrl+

# CHECK: beq- 2, target                  # encoding: [0x41,0xca,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beq- 2, target
# CHECK: beq- 0, target                  # encoding: [0x41,0xc2,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beq- target
# CHECK: beqa- 2, target                 # encoding: [0x41,0xca,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqa- 2, target
# CHECK: beqa- 0, target                 # encoding: [0x41,0xc2,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqa- target
# CHECK: beqlr- 2                        # encoding: [0x4d,0xca,0x00,0x20]
         beqlr- 2
# CHECK: beqlr- 0                        # encoding: [0x4d,0xc2,0x00,0x20]
         beqlr-
# CHECK: beqctr- 2                       # encoding: [0x4d,0xca,0x04,0x20]
         beqctr- 2
# CHECK: beqctr- 0                       # encoding: [0x4d,0xc2,0x04,0x20]
         beqctr-
# CHECK: beql- 2, target                 # encoding: [0x41,0xca,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beql- 2, target
# CHECK: beql- 0, target                 # encoding: [0x41,0xc2,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beql- target
# CHECK: beqla- 2, target                # encoding: [0x41,0xca,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqla- 2, target
# CHECK: beqla- 0, target                # encoding: [0x41,0xc2,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         beqla- target
# CHECK: beqlrl- 2                       # encoding: [0x4d,0xca,0x00,0x21]
         beqlrl- 2
# CHECK: beqlrl- 0                       # encoding: [0x4d,0xc2,0x00,0x21]
         beqlrl-
# CHECK: beqctrl- 2                      # encoding: [0x4d,0xca,0x04,0x21]
         beqctrl- 2
# CHECK: beqctrl- 0                      # encoding: [0x4d,0xc2,0x04,0x21]
         beqctrl-

# CHECK: bge 2, target                   # encoding: [0x40,0x88,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bge 2, target
# CHECK: bge 0, target                   # encoding: [0x40,0x80,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bge target
# CHECK: bgea 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgea 2, target
# CHECK: bgea 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgea target
# CHECK: bgelr 2                         # encoding: [0x4c,0x88,0x00,0x20]
         bgelr 2
# CHECK: bgelr 0                         # encoding: [0x4c,0x80,0x00,0x20]
         bgelr
# CHECK: bgectr 2                        # encoding: [0x4c,0x88,0x04,0x20]
         bgectr 2
# CHECK: bgectr 0                        # encoding: [0x4c,0x80,0x04,0x20]
         bgectr
# CHECK: bgel 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgel 2, target
# CHECK: bgel 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgel target
# CHECK: bgela 2, target                 # encoding: [0x40,0x88,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgela 2, target
# CHECK: bgela 0, target                 # encoding: [0x40,0x80,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgela target
# CHECK: bgelrl 2                        # encoding: [0x4c,0x88,0x00,0x21]
         bgelrl 2
# CHECK: bgelrl 0                        # encoding: [0x4c,0x80,0x00,0x21]
         bgelrl
# CHECK: bgectrl 2                       # encoding: [0x4c,0x88,0x04,0x21]
         bgectrl 2
# CHECK: bgectrl 0                       # encoding: [0x4c,0x80,0x04,0x21]
         bgectrl

# CHECK: bge+ 2, target                   # encoding: [0x40,0xe8,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bge+ 2, target
# CHECK: bge+ 0, target                   # encoding: [0x40,0xe0,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bge+ target
# CHECK: bgea+ 2, target                  # encoding: [0x40,0xe8,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgea+ 2, target
# CHECK: bgea+ 0, target                  # encoding: [0x40,0xe0,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgea+ target
# CHECK: bgelr+ 2                         # encoding: [0x4c,0xe8,0x00,0x20]
         bgelr+ 2
# CHECK: bgelr+ 0                         # encoding: [0x4c,0xe0,0x00,0x20]
         bgelr+
# CHECK: bgectr+ 2                        # encoding: [0x4c,0xe8,0x04,0x20]
         bgectr+ 2
# CHECK: bgectr+ 0                        # encoding: [0x4c,0xe0,0x04,0x20]
         bgectr+
# CHECK: bgel+ 2, target                  # encoding: [0x40,0xe8,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgel+ 2, target
# CHECK: bgel+ 0, target                  # encoding: [0x40,0xe0,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgel+ target
# CHECK: bgela+ 2, target                 # encoding: [0x40,0xe8,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgela+ 2, target
# CHECK: bgela+ 0, target                 # encoding: [0x40,0xe0,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgela+ target
# CHECK: bgelrl+ 2                        # encoding: [0x4c,0xe8,0x00,0x21]
         bgelrl+ 2
# CHECK: bgelrl+ 0                        # encoding: [0x4c,0xe0,0x00,0x21]
         bgelrl+
# CHECK: bgectrl+ 2                       # encoding: [0x4c,0xe8,0x04,0x21]
         bgectrl+ 2
# CHECK: bgectrl+ 0                       # encoding: [0x4c,0xe0,0x04,0x21]
         bgectrl+

# CHECK: bge- 2, target                   # encoding: [0x40,0xc8,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bge- 2, target
# CHECK: bge- 0, target                   # encoding: [0x40,0xc0,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bge- target
# CHECK: bgea- 2, target                  # encoding: [0x40,0xc8,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgea- 2, target
# CHECK: bgea- 0, target                  # encoding: [0x40,0xc0,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgea- target
# CHECK: bgelr- 2                         # encoding: [0x4c,0xc8,0x00,0x20]
         bgelr- 2
# CHECK: bgelr- 0                         # encoding: [0x4c,0xc0,0x00,0x20]
         bgelr-
# CHECK: bgectr- 2                        # encoding: [0x4c,0xc8,0x04,0x20]
         bgectr- 2
# CHECK: bgectr- 0                        # encoding: [0x4c,0xc0,0x04,0x20]
         bgectr-
# CHECK: bgel- 2, target                  # encoding: [0x40,0xc8,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgel- 2, target
# CHECK: bgel- 0, target                  # encoding: [0x40,0xc0,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgel- target
# CHECK: bgela- 2, target                 # encoding: [0x40,0xc8,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgela- 2, target
# CHECK: bgela- 0, target                 # encoding: [0x40,0xc0,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgela- target
# CHECK: bgelrl- 2                        # encoding: [0x4c,0xc8,0x00,0x21]
         bgelrl- 2
# CHECK: bgelrl- 0                        # encoding: [0x4c,0xc0,0x00,0x21]
         bgelrl-
# CHECK: bgectrl- 2                       # encoding: [0x4c,0xc8,0x04,0x21]
         bgectrl- 2
# CHECK: bgectrl- 0                       # encoding: [0x4c,0xc0,0x04,0x21]
         bgectrl-

# CHECK: bgt 2, target                   # encoding: [0x41,0x89,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgt 2, target
# CHECK: bgt 0, target                   # encoding: [0x41,0x81,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgt target
# CHECK: bgta 2, target                  # encoding: [0x41,0x89,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgta 2, target
# CHECK: bgta 0, target                  # encoding: [0x41,0x81,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgta target
# CHECK: bgtlr 2                         # encoding: [0x4d,0x89,0x00,0x20]
         bgtlr 2
# CHECK: bgtlr 0                         # encoding: [0x4d,0x81,0x00,0x20]
         bgtlr
# CHECK: bgtctr 2                        # encoding: [0x4d,0x89,0x04,0x20]
         bgtctr 2
# CHECK: bgtctr 0                        # encoding: [0x4d,0x81,0x04,0x20]
         bgtctr
# CHECK: bgtl 2, target                  # encoding: [0x41,0x89,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgtl 2, target
# CHECK: bgtl 0, target                  # encoding: [0x41,0x81,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgtl target
# CHECK: bgtla 2, target                 # encoding: [0x41,0x89,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgtla 2, target
# CHECK: bgtla 0, target                 # encoding: [0x41,0x81,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgtla target
# CHECK: bgtlrl 2                        # encoding: [0x4d,0x89,0x00,0x21]
         bgtlrl 2
# CHECK: bgtlrl 0                        # encoding: [0x4d,0x81,0x00,0x21]
         bgtlrl
# CHECK: bgtctrl 2                       # encoding: [0x4d,0x89,0x04,0x21]
         bgtctrl 2
# CHECK: bgtctrl 0                       # encoding: [0x4d,0x81,0x04,0x21]
         bgtctrl

# CHECK: bgt+ 2, target                  # encoding: [0x41,0xe9,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgt+ 2, target
# CHECK: bgt+ 0, target                  # encoding: [0x41,0xe1,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgt+ target
# CHECK: bgta+ 2, target                 # encoding: [0x41,0xe9,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgta+ 2, target
# CHECK: bgta+ 0, target                 # encoding: [0x41,0xe1,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgta+ target
# CHECK: bgtlr+ 2                        # encoding: [0x4d,0xe9,0x00,0x20]
         bgtlr+ 2
# CHECK: bgtlr+ 0                        # encoding: [0x4d,0xe1,0x00,0x20]
         bgtlr+
# CHECK: bgtctr+ 2                       # encoding: [0x4d,0xe9,0x04,0x20]
         bgtctr+ 2
# CHECK: bgtctr+ 0                       # encoding: [0x4d,0xe1,0x04,0x20]
         bgtctr+
# CHECK: bgtl+ 2, target                 # encoding: [0x41,0xe9,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgtl+ 2, target
# CHECK: bgtl+ 0, target                 # encoding: [0x41,0xe1,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgtl+ target
# CHECK: bgtla+ 2, target                # encoding: [0x41,0xe9,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgtla+ 2, target
# CHECK: bgtla+ 0, target                # encoding: [0x41,0xe1,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgtla+ target
# CHECK: bgtlrl+ 2                       # encoding: [0x4d,0xe9,0x00,0x21]
         bgtlrl+ 2
# CHECK: bgtlrl+ 0                       # encoding: [0x4d,0xe1,0x00,0x21]
         bgtlrl+
# CHECK: bgtctrl+ 2                      # encoding: [0x4d,0xe9,0x04,0x21]
         bgtctrl+ 2
# CHECK: bgtctrl+ 0                      # encoding: [0x4d,0xe1,0x04,0x21]
         bgtctrl+

# CHECK: bgt- 2, target                  # encoding: [0x41,0xc9,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgt- 2, target
# CHECK: bgt- 0, target                  # encoding: [0x41,0xc1,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgt- target
# CHECK: bgta- 2, target                 # encoding: [0x41,0xc9,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgta- 2, target
# CHECK: bgta- 0, target                 # encoding: [0x41,0xc1,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgta- target
# CHECK: bgtlr- 2                        # encoding: [0x4d,0xc9,0x00,0x20]
         bgtlr- 2
# CHECK: bgtlr- 0                        # encoding: [0x4d,0xc1,0x00,0x20]
         bgtlr-
# CHECK: bgtctr- 2                       # encoding: [0x4d,0xc9,0x04,0x20]
         bgtctr- 2
# CHECK: bgtctr- 0                       # encoding: [0x4d,0xc1,0x04,0x20]
         bgtctr-
# CHECK: bgtl- 2, target                 # encoding: [0x41,0xc9,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgtl- 2, target
# CHECK: bgtl- 0, target                 # encoding: [0x41,0xc1,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgtl- target
# CHECK: bgtla- 2, target                # encoding: [0x41,0xc9,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgtla- 2, target
# CHECK: bgtla- 0, target                # encoding: [0x41,0xc1,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bgtla- target
# CHECK: bgtlrl- 2                       # encoding: [0x4d,0xc9,0x00,0x21]
         bgtlrl- 2
# CHECK: bgtlrl- 0                       # encoding: [0x4d,0xc1,0x00,0x21]
         bgtlrl-
# CHECK: bgtctrl- 2                      # encoding: [0x4d,0xc9,0x04,0x21]
         bgtctrl- 2
# CHECK: bgtctrl- 0                      # encoding: [0x4d,0xc1,0x04,0x21]
         bgtctrl-

# CHECK: bge 2, target                   # encoding: [0x40,0x88,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnl 2, target
# CHECK: bge 0, target                   # encoding: [0x40,0x80,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnl target
# CHECK: bgea 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnla 2, target
# CHECK: bgea 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnla target
# CHECK: bgelr 2                         # encoding: [0x4c,0x88,0x00,0x20]
         bnllr 2
# CHECK: bgelr 0                         # encoding: [0x4c,0x80,0x00,0x20]
         bnllr
# CHECK: bgectr 2                        # encoding: [0x4c,0x88,0x04,0x20]
         bnlctr 2
# CHECK: bgectr 0                        # encoding: [0x4c,0x80,0x04,0x20]
         bnlctr
# CHECK: bgel 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnll 2, target
# CHECK: bgel 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnll target
# CHECK: bgela 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnlla 2, target
# CHECK: bgela 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnlla target
# CHECK: bgelrl 2                        # encoding: [0x4c,0x88,0x00,0x21]
         bnllrl 2
# CHECK: bgelrl 0                        # encoding: [0x4c,0x80,0x00,0x21]
         bnllrl
# CHECK: bgectrl 2                       # encoding: [0x4c,0x88,0x04,0x21]
         bnlctrl 2
# CHECK: bgectrl 0                       # encoding: [0x4c,0x80,0x04,0x21]
         bnlctrl

# CHECK: bge+ 2, target                  # encoding: [0x40,0xe8,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnl+ 2, target
# CHECK: bge+ 0, target                  # encoding: [0x40,0xe0,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnl+ target
# CHECK: bgea+ 2, target                 # encoding: [0x40,0xe8,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnla+ 2, target
# CHECK: bgea+ 0, target                 # encoding: [0x40,0xe0,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnla+ target
# CHECK: bgelr+ 2                        # encoding: [0x4c,0xe8,0x00,0x20]
         bnllr+ 2
# CHECK: bgelr+ 0                        # encoding: [0x4c,0xe0,0x00,0x20]
         bnllr+
# CHECK: bgectr+ 2                       # encoding: [0x4c,0xe8,0x04,0x20]
         bnlctr+ 2
# CHECK: bgectr+ 0                       # encoding: [0x4c,0xe0,0x04,0x20]
         bnlctr+
# CHECK: bgel+ 2, target                 # encoding: [0x40,0xe8,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnll+ 2, target
# CHECK: bgel+ 0, target                 # encoding: [0x40,0xe0,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnll+ target
# CHECK: bgela+ 2, target                # encoding: [0x40,0xe8,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnlla+ 2, target
# CHECK: bgela+ 0, target                # encoding: [0x40,0xe0,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnlla+ target
# CHECK: bgelrl+ 2                       # encoding: [0x4c,0xe8,0x00,0x21]
         bnllrl+ 2
# CHECK: bgelrl+ 0                       # encoding: [0x4c,0xe0,0x00,0x21]
         bnllrl+
# CHECK: bgectrl+ 2                      # encoding: [0x4c,0xe8,0x04,0x21]
         bnlctrl+ 2
# CHECK: bgectrl+ 0                      # encoding: [0x4c,0xe0,0x04,0x21]
         bnlctrl+

# CHECK: bge- 2, target                  # encoding: [0x40,0xc8,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnl- 2, target
# CHECK: bge- 0, target                  # encoding: [0x40,0xc0,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnl- target
# CHECK: bgea- 2, target                 # encoding: [0x40,0xc8,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnla- 2, target
# CHECK: bgea- 0, target                 # encoding: [0x40,0xc0,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnla- target
# CHECK: bgelr- 2                        # encoding: [0x4c,0xc8,0x00,0x20]
         bnllr- 2
# CHECK: bgelr- 0                        # encoding: [0x4c,0xc0,0x00,0x20]
         bnllr-
# CHECK: bgectr- 2                       # encoding: [0x4c,0xc8,0x04,0x20]
         bnlctr- 2
# CHECK: bgectr- 0                       # encoding: [0x4c,0xc0,0x04,0x20]
         bnlctr-
# CHECK: bgel- 2, target                 # encoding: [0x40,0xc8,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnll- 2, target
# CHECK: bgel- 0, target                 # encoding: [0x40,0xc0,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnll- target
# CHECK: bgela- 2, target                # encoding: [0x40,0xc8,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnlla- 2, target
# CHECK: bgela- 0, target                # encoding: [0x40,0xc0,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnlla- target
# CHECK: bgelrl- 2                       # encoding: [0x4c,0xc8,0x00,0x21]
         bnllrl- 2
# CHECK: bgelrl- 0                       # encoding: [0x4c,0xc0,0x00,0x21]
         bnllrl-
# CHECK: bgectrl- 2                      # encoding: [0x4c,0xc8,0x04,0x21]
         bnlctrl- 2
# CHECK: bgectrl- 0                      # encoding: [0x4c,0xc0,0x04,0x21]
         bnlctrl-

# CHECK: bne 2, target                   # encoding: [0x40,0x8a,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bne 2, target
# CHECK: bne 0, target                   # encoding: [0x40,0x82,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bne target
# CHECK: bnea 2, target                  # encoding: [0x40,0x8a,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnea 2, target
# CHECK: bnea 0, target                  # encoding: [0x40,0x82,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnea target
# CHECK: bnelr 2                         # encoding: [0x4c,0x8a,0x00,0x20]
         bnelr 2
# CHECK: bnelr 0                         # encoding: [0x4c,0x82,0x00,0x20]
         bnelr
# CHECK: bnectr 2                        # encoding: [0x4c,0x8a,0x04,0x20]
         bnectr 2
# CHECK: bnectr 0                        # encoding: [0x4c,0x82,0x04,0x20]
         bnectr
# CHECK: bnel 2, target                  # encoding: [0x40,0x8a,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnel 2, target
# CHECK: bnel 0, target                  # encoding: [0x40,0x82,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnel target
# CHECK: bnela 2, target                 # encoding: [0x40,0x8a,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnela 2, target
# CHECK: bnela 0, target                 # encoding: [0x40,0x82,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnela target
# CHECK: bnelrl 2                        # encoding: [0x4c,0x8a,0x00,0x21]
         bnelrl 2
# CHECK: bnelrl 0                        # encoding: [0x4c,0x82,0x00,0x21]
         bnelrl
# CHECK: bnectrl 2                       # encoding: [0x4c,0x8a,0x04,0x21]
         bnectrl 2
# CHECK: bnectrl 0                       # encoding: [0x4c,0x82,0x04,0x21]
         bnectrl

# CHECK: bne+ 2, target                  # encoding: [0x40,0xea,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bne+ 2, target
# CHECK: bne+ 0, target                  # encoding: [0x40,0xe2,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bne+ target
# CHECK: bnea+ 2, target                 # encoding: [0x40,0xea,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnea+ 2, target
# CHECK: bnea+ 0, target                 # encoding: [0x40,0xe2,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnea+ target
# CHECK: bnelr+ 2                        # encoding: [0x4c,0xea,0x00,0x20]
         bnelr+ 2
# CHECK: bnelr+ 0                        # encoding: [0x4c,0xe2,0x00,0x20]
         bnelr+
# CHECK: bnectr+ 2                       # encoding: [0x4c,0xea,0x04,0x20]
         bnectr+ 2
# CHECK: bnectr+ 0                       # encoding: [0x4c,0xe2,0x04,0x20]
         bnectr+
# CHECK: bnel+ 2, target                 # encoding: [0x40,0xea,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnel+ 2, target
# CHECK: bnel+ 0, target                 # encoding: [0x40,0xe2,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnel+ target
# CHECK: bnela+ 2, target                # encoding: [0x40,0xea,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnela+ 2, target
# CHECK: bnela+ 0, target                # encoding: [0x40,0xe2,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnela+ target
# CHECK: bnelrl+ 2                       # encoding: [0x4c,0xea,0x00,0x21]
         bnelrl+ 2
# CHECK: bnelrl+ 0                       # encoding: [0x4c,0xe2,0x00,0x21]
         bnelrl+
# CHECK: bnectrl+ 2                      # encoding: [0x4c,0xea,0x04,0x21]
         bnectrl+ 2
# CHECK: bnectrl+ 0                      # encoding: [0x4c,0xe2,0x04,0x21]
         bnectrl+

# CHECK: bne- 2, target                  # encoding: [0x40,0xca,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bne- 2, target
# CHECK: bne- 0, target                  # encoding: [0x40,0xc2,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bne- target
# CHECK: bnea- 2, target                 # encoding: [0x40,0xca,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnea- 2, target
# CHECK: bnea- 0, target                 # encoding: [0x40,0xc2,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnea- target
# CHECK: bnelr- 2                        # encoding: [0x4c,0xca,0x00,0x20]
         bnelr- 2
# CHECK: bnelr- 0                        # encoding: [0x4c,0xc2,0x00,0x20]
         bnelr-
# CHECK: bnectr- 2                       # encoding: [0x4c,0xca,0x04,0x20]
         bnectr- 2
# CHECK: bnectr- 0                       # encoding: [0x4c,0xc2,0x04,0x20]
         bnectr-
# CHECK: bnel- 2, target                 # encoding: [0x40,0xca,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnel- 2, target
# CHECK: bnel- 0, target                 # encoding: [0x40,0xc2,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnel- target
# CHECK: bnela- 2, target                # encoding: [0x40,0xca,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnela- 2, target
# CHECK: bnela- 0, target                # encoding: [0x40,0xc2,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnela- target
# CHECK: bnelrl- 2                       # encoding: [0x4c,0xca,0x00,0x21]
         bnelrl- 2
# CHECK: bnelrl- 0                       # encoding: [0x4c,0xc2,0x00,0x21]
         bnelrl-
# CHECK: bnectrl- 2                      # encoding: [0x4c,0xca,0x04,0x21]
         bnectrl- 2
# CHECK: bnectrl- 0                      # encoding: [0x4c,0xc2,0x04,0x21]
         bnectrl-

# CHECK: ble 2, target                   # encoding: [0x40,0x89,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bng 2, target
# CHECK: ble 0, target                   # encoding: [0x40,0x81,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bng target
# CHECK: blea 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnga 2, target
# CHECK: blea 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnga target
# CHECK: blelr 2                         # encoding: [0x4c,0x89,0x00,0x20]
         bnglr 2
# CHECK: blelr 0                         # encoding: [0x4c,0x81,0x00,0x20]
         bnglr
# CHECK: blectr 2                        # encoding: [0x4c,0x89,0x04,0x20]
         bngctr 2
# CHECK: blectr 0                        # encoding: [0x4c,0x81,0x04,0x20]
         bngctr
# CHECK: blel 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bngl 2, target
# CHECK: blel 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bngl target
# CHECK: blela 2, target                 # encoding: [0x40,0x89,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bngla 2, target
# CHECK: blela 0, target                 # encoding: [0x40,0x81,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bngla target
# CHECK: blelrl 2                        # encoding: [0x4c,0x89,0x00,0x21]
         bnglrl 2
# CHECK: blelrl 0                        # encoding: [0x4c,0x81,0x00,0x21]
         bnglrl
# CHECK: blectrl 2                       # encoding: [0x4c,0x89,0x04,0x21]
         bngctrl 2
# CHECK: blectrl 0                       # encoding: [0x4c,0x81,0x04,0x21]
         bngctrl

# CHECK: ble+ 2, target                   # encoding: [0x40,0xe9,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bng+ 2, target
# CHECK: ble+ 0, target                   # encoding: [0x40,0xe1,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bng+ target
# CHECK: blea+ 2, target                  # encoding: [0x40,0xe9,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnga+ 2, target
# CHECK: blea+ 0, target                  # encoding: [0x40,0xe1,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnga+ target
# CHECK: blelr+ 2                         # encoding: [0x4c,0xe9,0x00,0x20]
         bnglr+ 2
# CHECK: blelr+ 0                         # encoding: [0x4c,0xe1,0x00,0x20]
         bnglr+
# CHECK: blectr+ 2                        # encoding: [0x4c,0xe9,0x04,0x20]
         bngctr+ 2
# CHECK: blectr+ 0                        # encoding: [0x4c,0xe1,0x04,0x20]
         bngctr+
# CHECK: blel+ 2, target                  # encoding: [0x40,0xe9,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bngl+ 2, target
# CHECK: blel+ 0, target                  # encoding: [0x40,0xe1,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bngl+ target
# CHECK: blela+ 2, target                 # encoding: [0x40,0xe9,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bngla+ 2, target
# CHECK: blela+ 0, target                 # encoding: [0x40,0xe1,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bngla+ target
# CHECK: blelrl+ 2                        # encoding: [0x4c,0xe9,0x00,0x21]
         bnglrl+ 2
# CHECK: blelrl+ 0                        # encoding: [0x4c,0xe1,0x00,0x21]
         bnglrl+
# CHECK: blectrl+ 2                       # encoding: [0x4c,0xe9,0x04,0x21]
         bngctrl+ 2
# CHECK: blectrl+ 0                       # encoding: [0x4c,0xe1,0x04,0x21]
         bngctrl+

# CHECK: ble- 2, target                   # encoding: [0x40,0xc9,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bng- 2, target
# CHECK: ble- 0, target                   # encoding: [0x40,0xc1,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bng- target
# CHECK: blea- 2, target                  # encoding: [0x40,0xc9,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnga- 2, target
# CHECK: blea- 0, target                  # encoding: [0x40,0xc1,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnga- target
# CHECK: blelr- 2                         # encoding: [0x4c,0xc9,0x00,0x20]
         bnglr- 2
# CHECK: blelr- 0                         # encoding: [0x4c,0xc1,0x00,0x20]
         bnglr-
# CHECK: blectr- 2                        # encoding: [0x4c,0xc9,0x04,0x20]
         bngctr- 2
# CHECK: blectr- 0                        # encoding: [0x4c,0xc1,0x04,0x20]
         bngctr-
# CHECK: blel- 2, target                  # encoding: [0x40,0xc9,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bngl- 2, target
# CHECK: blel- 0, target                  # encoding: [0x40,0xc1,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bngl- target
# CHECK: blela- 2, target                 # encoding: [0x40,0xc9,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bngla- 2, target
# CHECK: blela- 0, target                 # encoding: [0x40,0xc1,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bngla- target
# CHECK: blelrl- 2                        # encoding: [0x4c,0xc9,0x00,0x21]
         bnglrl- 2
# CHECK: blelrl- 0                        # encoding: [0x4c,0xc1,0x00,0x21]
         bnglrl-
# CHECK: blectrl- 2                       # encoding: [0x4c,0xc9,0x04,0x21]
         bngctrl- 2
# CHECK: blectrl- 0                       # encoding: [0x4c,0xc1,0x04,0x21]
         bngctrl-

# CHECK: bun 2, target                   # encoding: [0x41,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bso 2, target
# CHECK: bun 0, target                   # encoding: [0x41,0x83,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bso target
# CHECK: buna 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsoa 2, target
# CHECK: buna 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsoa target
# CHECK: bunlr 2                         # encoding: [0x4d,0x8b,0x00,0x20]
         bsolr 2
# CHECK: bunlr 0                         # encoding: [0x4d,0x83,0x00,0x20]
         bsolr
# CHECK: bunctr 2                        # encoding: [0x4d,0x8b,0x04,0x20]
         bsoctr 2
# CHECK: bunctr 0                        # encoding: [0x4d,0x83,0x04,0x20]
         bsoctr
# CHECK: bunl 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bsol 2, target
# CHECK: bunl 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bsol target
# CHECK: bunla 2, target                 # encoding: [0x41,0x8b,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsola 2, target
# CHECK: bunla 0, target                 # encoding: [0x41,0x83,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsola target
# CHECK: bunlrl 2                        # encoding: [0x4d,0x8b,0x00,0x21]
         bsolrl 2
# CHECK: bunlrl 0                        # encoding: [0x4d,0x83,0x00,0x21]
         bsolrl
# CHECK: bunctrl 2                       # encoding: [0x4d,0x8b,0x04,0x21]
         bsoctrl 2
# CHECK: bunctrl 0                       # encoding: [0x4d,0x83,0x04,0x21]
         bsoctrl

# CHECK: bun+ 2, target                  # encoding: [0x41,0xeb,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bso+ 2, target
# CHECK: bun+ 0, target                  # encoding: [0x41,0xe3,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bso+ target
# CHECK: buna+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsoa+ 2, target
# CHECK: buna+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsoa+ target
# CHECK: bunlr+ 2                        # encoding: [0x4d,0xeb,0x00,0x20]
         bsolr+ 2
# CHECK: bunlr+ 0                        # encoding: [0x4d,0xe3,0x00,0x20]
         bsolr+
# CHECK: bunctr+ 2                       # encoding: [0x4d,0xeb,0x04,0x20]
         bsoctr+ 2
# CHECK: bunctr+ 0                       # encoding: [0x4d,0xe3,0x04,0x20]
         bsoctr+
# CHECK: bunl+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bsol+ 2, target
# CHECK: bunl+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bsol+ target
# CHECK: bunla+ 2, target                # encoding: [0x41,0xeb,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsola+ 2, target
# CHECK: bunla+ 0, target                # encoding: [0x41,0xe3,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsola+ target
# CHECK: bunlrl+ 2                       # encoding: [0x4d,0xeb,0x00,0x21]
         bsolrl+ 2
# CHECK: bunlrl+ 0                       # encoding: [0x4d,0xe3,0x00,0x21]
         bsolrl+
# CHECK: bunctrl+ 2                      # encoding: [0x4d,0xeb,0x04,0x21]
         bsoctrl+ 2
# CHECK: bunctrl+ 0                      # encoding: [0x4d,0xe3,0x04,0x21]
         bsoctrl+

# CHECK: bun- 2, target                  # encoding: [0x41,0xcb,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bso- 2, target
# CHECK: bun- 0, target                  # encoding: [0x41,0xc3,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bso- target
# CHECK: buna- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsoa- 2, target
# CHECK: buna- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsoa- target
# CHECK: bunlr- 2                        # encoding: [0x4d,0xcb,0x00,0x20]
         bsolr- 2
# CHECK: bunlr- 0                        # encoding: [0x4d,0xc3,0x00,0x20]
         bsolr-
# CHECK: bunctr- 2                       # encoding: [0x4d,0xcb,0x04,0x20]
         bsoctr- 2
# CHECK: bunctr- 0                       # encoding: [0x4d,0xc3,0x04,0x20]
         bsoctr-
# CHECK: bunl- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bsol- 2, target
# CHECK: bunl- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bsol- target
# CHECK: bunla- 2, target                # encoding: [0x41,0xcb,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsola- 2, target
# CHECK: bunla- 0, target                # encoding: [0x41,0xc3,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bsola- target
# CHECK: bunlrl- 2                       # encoding: [0x4d,0xcb,0x00,0x21]
         bsolrl- 2
# CHECK: bunlrl- 0                       # encoding: [0x4d,0xc3,0x00,0x21]
         bsolrl-
# CHECK: bunctrl- 2                      # encoding: [0x4d,0xcb,0x04,0x21]
         bsoctrl- 2
# CHECK: bunctrl- 0                      # encoding: [0x4d,0xc3,0x04,0x21]
         bsoctrl-

# CHECK: bnu 2, target                   # encoding: [0x40,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bns 2, target
# CHECK: bnu 0, target                   # encoding: [0x40,0x83,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bns target
# CHECK: bnua 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsa 2, target
# CHECK: bnua 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsa target
# CHECK: bnulr 2                         # encoding: [0x4c,0x8b,0x00,0x20]
         bnslr 2
# CHECK: bnulr 0                         # encoding: [0x4c,0x83,0x00,0x20]
         bnslr
# CHECK: bnuctr 2                        # encoding: [0x4c,0x8b,0x04,0x20]
         bnsctr 2
# CHECK: bnuctr 0                        # encoding: [0x4c,0x83,0x04,0x20]
         bnsctr
# CHECK: bnul 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnsl 2, target
# CHECK: bnul 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnsl target
# CHECK: bnula 2, target                 # encoding: [0x40,0x8b,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsla 2, target
# CHECK: bnula 0, target                 # encoding: [0x40,0x83,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsla target
# CHECK: bnulrl 2                        # encoding: [0x4c,0x8b,0x00,0x21]
         bnslrl 2
# CHECK: bnulrl 0                        # encoding: [0x4c,0x83,0x00,0x21]
         bnslrl
# CHECK: bnuctrl 2                       # encoding: [0x4c,0x8b,0x04,0x21]
         bnsctrl 2
# CHECK: bnuctrl 0                       # encoding: [0x4c,0x83,0x04,0x21]
         bnsctrl

# CHECK: bnu+ 2, target                  # encoding: [0x40,0xeb,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bns+ 2, target
# CHECK: bnu+ 0, target                  # encoding: [0x40,0xe3,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bns+ target
# CHECK: bnua+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsa+ 2, target
# CHECK: bnua+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsa+ target
# CHECK: bnulr+ 2                        # encoding: [0x4c,0xeb,0x00,0x20]
         bnslr+ 2
# CHECK: bnulr+ 0                        # encoding: [0x4c,0xe3,0x00,0x20]
         bnslr+
# CHECK: bnuctr+ 2                       # encoding: [0x4c,0xeb,0x04,0x20]
         bnsctr+ 2
# CHECK: bnuctr+ 0                       # encoding: [0x4c,0xe3,0x04,0x20]
         bnsctr+
# CHECK: bnul+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnsl+ 2, target
# CHECK: bnul+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnsl+ target
# CHECK: bnula+ 2, target                # encoding: [0x40,0xeb,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsla+ 2, target
# CHECK: bnula+ 0, target                # encoding: [0x40,0xe3,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsla+ target
# CHECK: bnulrl+ 2                       # encoding: [0x4c,0xeb,0x00,0x21]
         bnslrl+ 2
# CHECK: bnulrl+ 0                       # encoding: [0x4c,0xe3,0x00,0x21]
         bnslrl+
# CHECK: bnuctrl+ 2                      # encoding: [0x4c,0xeb,0x04,0x21]
         bnsctrl+ 2
# CHECK: bnuctrl+ 0                      # encoding: [0x4c,0xe3,0x04,0x21]
         bnsctrl+

# CHECK: bnu- 2, target                  # encoding: [0x40,0xcb,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bns- 2, target
# CHECK: bnu- 0, target                  # encoding: [0x40,0xc3,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bns- target
# CHECK: bnua- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsa- 2, target
# CHECK: bnua- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsa- target
# CHECK: bnulr- 2                        # encoding: [0x4c,0xcb,0x00,0x20]
         bnslr- 2
# CHECK: bnulr- 0                        # encoding: [0x4c,0xc3,0x00,0x20]
         bnslr-
# CHECK: bnuctr- 2                       # encoding: [0x4c,0xcb,0x04,0x20]
         bnsctr- 2
# CHECK: bnuctr- 0                       # encoding: [0x4c,0xc3,0x04,0x20]
         bnsctr-
# CHECK: bnul- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnsl- 2, target
# CHECK: bnul- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnsl- target
# CHECK: bnula- 2, target                # encoding: [0x40,0xcb,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsla- 2, target
# CHECK: bnula- 0, target                # encoding: [0x40,0xc3,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnsla- target
# CHECK: bnulrl- 2                       # encoding: [0x4c,0xcb,0x00,0x21]
         bnslrl- 2
# CHECK: bnulrl- 0                       # encoding: [0x4c,0xc3,0x00,0x21]
         bnslrl-
# CHECK: bnuctrl- 2                      # encoding: [0x4c,0xcb,0x04,0x21]
         bnsctrl- 2
# CHECK: bnuctrl- 0                      # encoding: [0x4c,0xc3,0x04,0x21]
         bnsctrl-

# CHECK: bun 2, target                   # encoding: [0x41,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bun 2, target
# CHECK: bun 0, target                   # encoding: [0x41,0x83,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bun target
# CHECK: buna 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         buna 2, target
# CHECK: buna 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         buna target
# CHECK: bunlr 2                         # encoding: [0x4d,0x8b,0x00,0x20]
         bunlr 2
# CHECK: bunlr 0                         # encoding: [0x4d,0x83,0x00,0x20]
         bunlr
# CHECK: bunctr 2                        # encoding: [0x4d,0x8b,0x04,0x20]
         bunctr 2
# CHECK: bunctr 0                        # encoding: [0x4d,0x83,0x04,0x20]
         bunctr
# CHECK: bunl 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bunl 2, target
# CHECK: bunl 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bunl target
# CHECK: bunla 2, target                 # encoding: [0x41,0x8b,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bunla 2, target
# CHECK: bunla 0, target                 # encoding: [0x41,0x83,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bunla target
# CHECK: bunlrl 2                        # encoding: [0x4d,0x8b,0x00,0x21]
         bunlrl 2
# CHECK: bunlrl 0                        # encoding: [0x4d,0x83,0x00,0x21]
         bunlrl
# CHECK: bunctrl 2                       # encoding: [0x4d,0x8b,0x04,0x21]
         bunctrl 2
# CHECK: bunctrl 0                       # encoding: [0x4d,0x83,0x04,0x21]
         bunctrl

# CHECK: bun+ 2, target                  # encoding: [0x41,0xeb,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bun+ 2, target
# CHECK: bun+ 0, target                  # encoding: [0x41,0xe3,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bun+ target
# CHECK: buna+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         buna+ 2, target
# CHECK: buna+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         buna+ target
# CHECK: bunlr+ 2                        # encoding: [0x4d,0xeb,0x00,0x20]
         bunlr+ 2
# CHECK: bunlr+ 0                        # encoding: [0x4d,0xe3,0x00,0x20]
         bunlr+
# CHECK: bunctr+ 2                       # encoding: [0x4d,0xeb,0x04,0x20]
         bunctr+ 2
# CHECK: bunctr+ 0                       # encoding: [0x4d,0xe3,0x04,0x20]
         bunctr+
# CHECK: bunl+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bunl+ 2, target
# CHECK: bunl+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bunl+ target
# CHECK: bunla+ 2, target                # encoding: [0x41,0xeb,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bunla+ 2, target
# CHECK: bunla+ 0, target                # encoding: [0x41,0xe3,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bunla+ target
# CHECK: bunlrl+ 2                       # encoding: [0x4d,0xeb,0x00,0x21]
         bunlrl+ 2
# CHECK: bunlrl+ 0                       # encoding: [0x4d,0xe3,0x00,0x21]
         bunlrl+
# CHECK: bunctrl+ 2                      # encoding: [0x4d,0xeb,0x04,0x21]
         bunctrl+ 2
# CHECK: bunctrl+ 0                      # encoding: [0x4d,0xe3,0x04,0x21]
         bunctrl+

# CHECK: bun- 2, target                  # encoding: [0x41,0xcb,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bun- 2, target
# CHECK: bun- 0, target                  # encoding: [0x41,0xc3,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bun- target
# CHECK: buna- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         buna- 2, target
# CHECK: buna- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         buna- target
# CHECK: bunlr- 2                        # encoding: [0x4d,0xcb,0x00,0x20]
         bunlr- 2
# CHECK: bunlr- 0                        # encoding: [0x4d,0xc3,0x00,0x20]
         bunlr-
# CHECK: bunctr- 2                       # encoding: [0x4d,0xcb,0x04,0x20]
         bunctr- 2
# CHECK: bunctr- 0                       # encoding: [0x4d,0xc3,0x04,0x20]
         bunctr-
# CHECK: bunl- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bunl- 2, target
# CHECK: bunl- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bunl- target
# CHECK: bunla- 2, target                # encoding: [0x41,0xcb,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bunla- 2, target
# CHECK: bunla- 0, target                # encoding: [0x41,0xc3,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bunla- target
# CHECK: bunlrl- 2                       # encoding: [0x4d,0xcb,0x00,0x21]
         bunlrl- 2
# CHECK: bunlrl- 0                       # encoding: [0x4d,0xc3,0x00,0x21]
         bunlrl-
# CHECK: bunctrl- 2                      # encoding: [0x4d,0xcb,0x04,0x21]
         bunctrl- 2
# CHECK: bunctrl- 0                      # encoding: [0x4d,0xc3,0x04,0x21]
         bunctrl-

# CHECK: bnu 2, target                   # encoding: [0x40,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnu 2, target
# CHECK: bnu 0, target                   # encoding: [0x40,0x83,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnu target
# CHECK: bnua 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnua 2, target
# CHECK: bnua 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnua target
# CHECK: bnulr 2                         # encoding: [0x4c,0x8b,0x00,0x20]
         bnulr 2
# CHECK: bnulr 0                         # encoding: [0x4c,0x83,0x00,0x20]
         bnulr
# CHECK: bnuctr 2                        # encoding: [0x4c,0x8b,0x04,0x20]
         bnuctr 2
# CHECK: bnuctr 0                        # encoding: [0x4c,0x83,0x04,0x20]
         bnuctr
# CHECK: bnul 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnul 2, target
# CHECK: bnul 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnul target
# CHECK: bnula 2, target                 # encoding: [0x40,0x8b,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnula 2, target
# CHECK: bnula 0, target                 # encoding: [0x40,0x83,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnula target
# CHECK: bnulrl 2                        # encoding: [0x4c,0x8b,0x00,0x21]
         bnulrl 2
# CHECK: bnulrl 0                        # encoding: [0x4c,0x83,0x00,0x21]
         bnulrl
# CHECK: bnuctrl 2                       # encoding: [0x4c,0x8b,0x04,0x21]
         bnuctrl 2
# CHECK: bnuctrl 0                       # encoding: [0x4c,0x83,0x04,0x21]
         bnuctrl

# CHECK: bnu+ 2, target                  # encoding: [0x40,0xeb,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnu+ 2, target
# CHECK: bnu+ 0, target                  # encoding: [0x40,0xe3,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnu+ target
# CHECK: bnua+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnua+ 2, target
# CHECK: bnua+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnua+ target
# CHECK: bnulr+ 2                        # encoding: [0x4c,0xeb,0x00,0x20]
         bnulr+ 2
# CHECK: bnulr+ 0                        # encoding: [0x4c,0xe3,0x00,0x20]
         bnulr+
# CHECK: bnuctr+ 2                       # encoding: [0x4c,0xeb,0x04,0x20]
         bnuctr+ 2
# CHECK: bnuctr+ 0                       # encoding: [0x4c,0xe3,0x04,0x20]
         bnuctr+
# CHECK: bnul+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnul+ 2, target
# CHECK: bnul+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnul+ target
# CHECK: bnula+ 2, target                # encoding: [0x40,0xeb,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnula+ 2, target
# CHECK: bnula+ 0, target                # encoding: [0x40,0xe3,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnula+ target
# CHECK: bnulrl+ 2                       # encoding: [0x4c,0xeb,0x00,0x21]
         bnulrl+ 2
# CHECK: bnulrl+ 0                       # encoding: [0x4c,0xe3,0x00,0x21]
         bnulrl+
# CHECK: bnuctrl+ 2                      # encoding: [0x4c,0xeb,0x04,0x21]
         bnuctrl+ 2
# CHECK: bnuctrl+ 0                      # encoding: [0x4c,0xe3,0x04,0x21]
         bnuctrl+

# CHECK: bnu- 2, target                  # encoding: [0x40,0xcb,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnu- 2, target
# CHECK: bnu- 0, target                  # encoding: [0x40,0xc3,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnu- target
# CHECK: bnua- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnua- 2, target
# CHECK: bnua- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnua- target
# CHECK: bnulr- 2                        # encoding: [0x4c,0xcb,0x00,0x20]
         bnulr- 2
# CHECK: bnulr- 0                        # encoding: [0x4c,0xc3,0x00,0x20]
         bnulr-
# CHECK: bnuctr- 2                       # encoding: [0x4c,0xcb,0x04,0x20]
         bnuctr- 2
# CHECK: bnuctr- 0                       # encoding: [0x4c,0xc3,0x04,0x20]
         bnuctr-
# CHECK: bnul- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnul- 2, target
# CHECK: bnul- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnul- target
# CHECK: bnula- 2, target                # encoding: [0x40,0xcb,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnula- 2, target
# CHECK: bnula- 0, target                # encoding: [0x40,0xc3,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bnula- target
# CHECK: bnulrl- 2                       # encoding: [0x4c,0xcb,0x00,0x21]
         bnulrl- 2
# CHECK: bnulrl- 0                       # encoding: [0x4c,0xc3,0x00,0x21]
         bnulrl-
# CHECK: bnuctrl- 2                      # encoding: [0x4c,0xcb,0x04,0x21]
         bnuctrl- 2
# CHECK: bnuctrl- 0                      # encoding: [0x4c,0xc3,0x04,0x21]
         bnuctrl-

# Condition register logical mnemonics

# CHECK: creqv 2, 2, 2                   # encoding: [0x4c,0x42,0x12,0x42]
         crset 2
# CHECK: crxor 2, 2, 2                   # encoding: [0x4c,0x42,0x11,0x82]
         crclr 2
# CHECK: cror 2, 3, 3                    # encoding: [0x4c,0x43,0x1b,0x82]
         crmove 2, 3
# CHECK: crnor 2, 3, 3                   # encoding: [0x4c,0x43,0x18,0x42]
         crnot 2, 3

# Subtract mnemonics

# CHECK: addi 2, 3, -128                 # encoding: [0x38,0x43,0xff,0x80]
         subi 2, 3, 128
# CHECK: addis 2, 3, -128                # encoding: [0x3c,0x43,0xff,0x80]
         subis 2, 3, 128
# CHECK: addic 2, 3, -128                # encoding: [0x30,0x43,0xff,0x80]
         subic 2, 3, 128
# CHECK: addic. 2, 3, -128               # encoding: [0x34,0x43,0xff,0x80]
         subic. 2, 3, 128

# CHECK: subf 2, 4, 3                    # encoding: [0x7c,0x44,0x18,0x50]
         sub 2, 3, 4
# CHECK: subf. 2, 4, 3                   # encoding: [0x7c,0x44,0x18,0x51]
         sub. 2, 3, 4
# CHECK: subfc 2, 4, 3                   # encoding: [0x7c,0x44,0x18,0x10]
         subc 2, 3, 4
# CHECK: subfc. 2, 4, 3                  # encoding: [0x7c,0x44,0x18,0x11]
         subc. 2, 3, 4

# Compare mnemonics

# CHECK: cmpdi 2, 3, 128                 # encoding: [0x2d,0x23,0x00,0x80]
         cmpdi 2, 3, 128
# CHECK: cmpdi 0, 3, 128                 # encoding: [0x2c,0x23,0x00,0x80]
         cmpdi 3, 128
# CHECK: cmpd 2, 3, 4                    # encoding: [0x7d,0x23,0x20,0x00]
         cmpd 2, 3, 4
# CHECK: cmpd 0, 3, 4                    # encoding: [0x7c,0x23,0x20,0x00]
         cmpd 3, 4
# CHECK: cmpldi 2, 3, 128                # encoding: [0x29,0x23,0x00,0x80]
         cmpldi 2, 3, 128
# CHECK: cmpldi 0, 3, 128                # encoding: [0x28,0x23,0x00,0x80]
         cmpldi 3, 128
# CHECK: cmpld 2, 3, 4                   # encoding: [0x7d,0x23,0x20,0x40]
         cmpld 2, 3, 4
# CHECK: cmpld 0, 3, 4                   # encoding: [0x7c,0x23,0x20,0x40]
         cmpld 3, 4

# CHECK: cmpwi 2, 3, 128                 # encoding: [0x2d,0x03,0x00,0x80]
         cmpwi 2, 3, 128
# CHECK: cmpwi 0, 3, 128                 # encoding: [0x2c,0x03,0x00,0x80]
         cmpwi 3, 128
# CHECK: cmpw 2, 3, 4                    # encoding: [0x7d,0x03,0x20,0x00]
         cmpw 2, 3, 4
# CHECK: cmpw 0, 3, 4                    # encoding: [0x7c,0x03,0x20,0x00]
         cmpw 3, 4
# CHECK: cmplwi 2, 3, 128                # encoding: [0x29,0x03,0x00,0x80]
         cmplwi 2, 3, 128
# CHECK: cmplwi 0, 3, 128                # encoding: [0x28,0x03,0x00,0x80]
         cmplwi 3, 128
# CHECK: cmplw 2, 3, 4                   # encoding: [0x7d,0x03,0x20,0x40]
         cmplw 2, 3, 4
# CHECK: cmplw 0, 3, 4                   # encoding: [0x7c,0x03,0x20,0x40]
         cmplw 3, 4

# Trap mnemonics

# CHECK: twi 16, 3, 4                    # encoding: [0x0e,0x03,0x00,0x04]
         twlti 3, 4
# CHECK: tw 16, 3, 4                     # encoding: [0x7e,0x03,0x20,0x08]
         twlt 3, 4
# CHECK: tdi 16, 3, 4                    # encoding: [0x0a,0x03,0x00,0x04]
         tdlti 3, 4
# CHECK: td 16, 3, 4                     # encoding: [0x7e,0x03,0x20,0x88]
         tdlt 3, 4

# CHECK: twi 20, 3, 4                    # encoding: [0x0e,0x83,0x00,0x04]
         twlei 3, 4
# CHECK: tw 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x08]
         twle 3, 4
# CHECK: tdi 20, 3, 4                    # encoding: [0x0a,0x83,0x00,0x04]
         tdlei 3, 4
# CHECK: td 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x88]
         tdle 3, 4

# CHECK: twi 4, 3, 4                     # encoding: [0x0c,0x83,0x00,0x04]
         tweqi 3, 4
# CHECK: tw 4, 3, 4                      # encoding: [0x7c,0x83,0x20,0x08]
         tweq 3, 4
# CHECK: tdi 4, 3, 4                     # encoding: [0x08,0x83,0x00,0x04]
         tdeqi 3, 4
# CHECK: td 4, 3, 4                      # encoding: [0x7c,0x83,0x20,0x88]
         tdeq 3, 4

# CHECK: twi 12, 3, 4                    # encoding: [0x0d,0x83,0x00,0x04]
         twgei 3, 4
# CHECK: tw 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x08]
         twge 3, 4
# CHECK: tdi 12, 3, 4                    # encoding: [0x09,0x83,0x00,0x04]
         tdgei 3, 4
# CHECK: td 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x88]
         tdge 3, 4

# CHECK: twi 8, 3, 4                     # encoding: [0x0d,0x03,0x00,0x04]
         twgti 3, 4
# CHECK: tw 8, 3, 4                      # encoding: [0x7d,0x03,0x20,0x08]
         twgt 3, 4
# CHECK: tdi 8, 3, 4                     # encoding: [0x09,0x03,0x00,0x04]
         tdgti 3, 4
# CHECK: td 8, 3, 4                      # encoding: [0x7d,0x03,0x20,0x88]
         tdgt 3, 4

# CHECK: twi 12, 3, 4                    # encoding: [0x0d,0x83,0x00,0x04]
         twnli 3, 4
# CHECK: tw 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x08]
         twnl 3, 4
# CHECK: tdi 12, 3, 4                    # encoding: [0x09,0x83,0x00,0x04]
         tdnli 3, 4
# CHECK: td 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x88]
         tdnl 3, 4

# CHECK: twi 24, 3, 4                    # encoding: [0x0f,0x03,0x00,0x04]
         twnei 3, 4
# CHECK: tw 24, 3, 4                     # encoding: [0x7f,0x03,0x20,0x08]
         twne 3, 4
# CHECK: tdi 24, 3, 4                    # encoding: [0x0b,0x03,0x00,0x04]
         tdnei 3, 4
# CHECK: td 24, 3, 4                     # encoding: [0x7f,0x03,0x20,0x88]
         tdne 3, 4

# CHECK: twi 20, 3, 4                    # encoding: [0x0e,0x83,0x00,0x04]
         twngi 3, 4
# CHECK: tw 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x08]
         twng 3, 4
# CHECK: tdi 20, 3, 4                    # encoding: [0x0a,0x83,0x00,0x04]
         tdngi 3, 4
# CHECK: td 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x88]
         tdng 3, 4

# CHECK: twi 2, 3, 4                     # encoding: [0x0c,0x43,0x00,0x04]
         twllti 3, 4
# CHECK: tw 2, 3, 4                      # encoding: [0x7c,0x43,0x20,0x08]
         twllt 3, 4
# CHECK: tdi 2, 3, 4                     # encoding: [0x08,0x43,0x00,0x04]
         tdllti 3, 4
# CHECK: td 2, 3, 4                      # encoding: [0x7c,0x43,0x20,0x88]
         tdllt 3, 4

# CHECK: twi 6, 3, 4                     # encoding: [0x0c,0xc3,0x00,0x04]
         twllei 3, 4
# CHECK: tw 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x08]
         twlle 3, 4
# CHECK: tdi 6, 3, 4                     # encoding: [0x08,0xc3,0x00,0x04]
         tdllei 3, 4
# CHECK: td 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x88]
         tdlle 3, 4

# CHECK: twi 5, 3, 4                     # encoding: [0x0c,0xa3,0x00,0x04]
         twlgei 3, 4
# CHECK: tw 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x08]
         twlge 3, 4
# CHECK: tdi 5, 3, 4                     # encoding: [0x08,0xa3,0x00,0x04]
         tdlgei 3, 4
# CHECK: td 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x88]
         tdlge 3, 4

# CHECK: twi 1, 3, 4                     # encoding: [0x0c,0x23,0x00,0x04]
         twlgti 3, 4
# CHECK: tw 1, 3, 4                      # encoding: [0x7c,0x23,0x20,0x08]
         twlgt 3, 4
# CHECK: tdi 1, 3, 4                     # encoding: [0x08,0x23,0x00,0x04]
         tdlgti 3, 4
# CHECK: td 1, 3, 4                      # encoding: [0x7c,0x23,0x20,0x88]
         tdlgt 3, 4

# CHECK: twi 5, 3, 4                     # encoding: [0x0c,0xa3,0x00,0x04]
         twlnli 3, 4
# CHECK: tw 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x08]
         twlnl 3, 4
# CHECK: tdi 5, 3, 4                     # encoding: [0x08,0xa3,0x00,0x04]
         tdlnli 3, 4
# CHECK: td 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x88]
         tdlnl 3, 4

# CHECK: twi 6, 3, 4                     # encoding: [0x0c,0xc3,0x00,0x04]
         twlngi 3, 4
# CHECK: tw 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x08]
         twlng 3, 4
# CHECK: tdi 6, 3, 4                     # encoding: [0x08,0xc3,0x00,0x04]
         tdlngi 3, 4
# CHECK: td 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x88]
         tdlng 3, 4

# CHECK: twi 31, 3, 4                    # encoding: [0x0f,0xe3,0x00,0x04]
         twui 3, 4
# CHECK: tw 31, 3, 4                     # encoding: [0x7f,0xe3,0x20,0x08]
         twu 3, 4
# CHECK: tdi 31, 3, 4                    # encoding: [0x0b,0xe3,0x00,0x04]
         tdui 3, 4
# CHECK: td 31, 3, 4                     # encoding: [0x7f,0xe3,0x20,0x88]
         tdu 3, 4

# CHECK: trap                            # encoding: [0x7f,0xe0,0x00,0x08]
         trap

# Rotate and shift mnemonics

# CHECK: rldicr 2, 3, 5, 3               # encoding: [0x78,0x62,0x28,0xc4]
         extldi 2, 3, 4, 5
# CHECK: rldicr. 2, 3, 5, 3              # encoding: [0x78,0x62,0x28,0xc5]
         extldi. 2, 3, 4, 5
# CHECK: rldicl 2, 3, 9, 60              # encoding: [0x78,0x62,0x4f,0x20]
         extrdi 2, 3, 4, 5
# CHECK: rldicl. 2, 3, 9, 60             # encoding: [0x78,0x62,0x4f,0x21]
         extrdi. 2, 3, 4, 5
# CHECK: rldimi 2, 3, 55, 5              # encoding: [0x78,0x62,0xb9,0x4e]
         insrdi 2, 3, 4, 5
# CHECK: rldimi. 2, 3, 55, 5             # encoding: [0x78,0x62,0xb9,0x4f]
         insrdi. 2, 3, 4, 5
# CHECK: rldicl 2, 3, 4, 0               # encoding: [0x78,0x62,0x20,0x00]
         rotldi 2, 3, 4
# CHECK: rldicl. 2, 3, 4, 0              # encoding: [0x78,0x62,0x20,0x01]
         rotldi. 2, 3, 4
# CHECK: rldicl 2, 3, 60, 0              # encoding: [0x78,0x62,0xe0,0x02]
         rotrdi 2, 3, 4
# CHECK: rldicl. 2, 3, 60, 0             # encoding: [0x78,0x62,0xe0,0x03]
         rotrdi. 2, 3, 4
# CHECK: rldcl 2, 3, 4, 0                # encoding: [0x78,0x62,0x20,0x10]
         rotld 2, 3, 4
# CHECK: rldcl. 2, 3, 4, 0               # encoding: [0x78,0x62,0x20,0x11]
         rotld. 2, 3, 4
# CHECK: sldi 2, 3, 4                    # encoding: [0x78,0x62,0x26,0xe4]
         sldi 2, 3, 4
# CHECK: rldicr. 2, 3, 4, 59             # encoding: [0x78,0x62,0x26,0xe5]
         sldi. 2, 3, 4
# CHECK: rldicl 2, 3, 60, 4              # encoding: [0x78,0x62,0xe1,0x02]
         srdi 2, 3, 4
# CHECK: rldicl. 2, 3, 60, 4             # encoding: [0x78,0x62,0xe1,0x03]
         srdi. 2, 3, 4
# CHECK: rldicl 2, 3, 0, 4               # encoding: [0x78,0x62,0x01,0x00]
         clrldi 2, 3, 4
# CHECK: rldicl. 2, 3, 0, 4              # encoding: [0x78,0x62,0x01,0x01]
         clrldi. 2, 3, 4
# CHECK: rldicr 2, 3, 0, 59              # encoding: [0x78,0x62,0x06,0xe4]
         clrrdi 2, 3, 4
# CHECK: rldicr. 2, 3, 0, 59             # encoding: [0x78,0x62,0x06,0xe5]
         clrrdi. 2, 3, 4
# CHECK: rldic 2, 3, 4, 1                # encoding: [0x78,0x62,0x20,0x48]
         clrlsldi 2, 3, 5, 4
# CHECK: rldic. 2, 3, 4, 1               # encoding: [0x78,0x62,0x20,0x49]
         clrlsldi. 2, 3, 5, 4

# CHECK: rlwinm 2, 3, 5, 0, 3            # encoding: [0x54,0x62,0x28,0x06]
         extlwi 2, 3, 4, 5
# CHECK: rlwinm. 2, 3, 5, 0, 3           # encoding: [0x54,0x62,0x28,0x07]
         extlwi. 2, 3, 4, 5
# CHECK: rlwinm 2, 3, 9, 28, 31          # encoding: [0x54,0x62,0x4f,0x3e]
         extrwi 2, 3, 4, 5
# CHECK: rlwinm. 2, 3, 9, 28, 31         # encoding: [0x54,0x62,0x4f,0x3f]
         extrwi. 2, 3, 4, 5
# CHECK: rlwimi 2, 3, 27, 5, 8           # encoding: [0x50,0x62,0xd9,0x50]
         inslwi 2, 3, 4, 5
# CHECK: rlwimi. 2, 3, 27, 5, 8          # encoding: [0x50,0x62,0xd9,0x51]
         inslwi. 2, 3, 4, 5
# CHECK: rlwimi 2, 3, 23, 5, 8           # encoding: [0x50,0x62,0xb9,0x50]
         insrwi 2, 3, 4, 5
# CHECK: rlwimi. 2, 3, 23, 5, 8          # encoding: [0x50,0x62,0xb9,0x51]
         insrwi. 2, 3, 4, 5
# CHECK: rlwinm 2, 3, 4, 0, 31           # encoding: [0x54,0x62,0x20,0x3e]
         rotlwi 2, 3, 4
# CHECK: rlwinm. 2, 3, 4, 0, 31          # encoding: [0x54,0x62,0x20,0x3f]
         rotlwi. 2, 3, 4
# CHECK: rlwinm 2, 3, 28, 0, 31          # encoding: [0x54,0x62,0xe0,0x3e]
         rotrwi 2, 3, 4
# CHECK: rlwinm. 2, 3, 28, 0, 31         # encoding: [0x54,0x62,0xe0,0x3f]
         rotrwi. 2, 3, 4
# CHECK: rlwnm 2, 3, 4, 0, 31            # encoding: [0x5c,0x62,0x20,0x3e]
         rotlw 2, 3, 4
# CHECK: rlwnm. 2, 3, 4, 0, 31           # encoding: [0x5c,0x62,0x20,0x3f]
         rotlw. 2, 3, 4
# CHECK: slwi 2, 3, 4                    # encoding: [0x54,0x62,0x20,0x36]
         slwi 2, 3, 4
# CHECK: rlwinm. 2, 3, 4, 0, 27          # encoding: [0x54,0x62,0x20,0x37]
         slwi. 2, 3, 4
# CHECK: srwi 2, 3, 4                    # encoding: [0x54,0x62,0xe1,0x3e]
         srwi 2, 3, 4
# CHECK: rlwinm. 2, 3, 28, 4, 31         # encoding: [0x54,0x62,0xe1,0x3f]
         srwi. 2, 3, 4
# CHECK: rlwinm 2, 3, 0, 4, 31           # encoding: [0x54,0x62,0x01,0x3e]
         clrlwi 2, 3, 4
# CHECK: rlwinm. 2, 3, 0, 4, 31          # encoding: [0x54,0x62,0x01,0x3f]
         clrlwi. 2, 3, 4
# CHECK: rlwinm 2, 3, 0, 0, 27           # encoding: [0x54,0x62,0x00,0x36]
         clrrwi 2, 3, 4
# CHECK: rlwinm. 2, 3, 0, 0, 27          # encoding: [0x54,0x62,0x00,0x37]
         clrrwi. 2, 3, 4
# CHECK: rlwinm 2, 3, 4, 1, 27           # encoding: [0x54,0x62,0x20,0x76]
         clrlslwi 2, 3, 5, 4
# CHECK: rlwinm. 2, 3, 4, 1, 27          # encoding: [0x54,0x62,0x20,0x77]
         clrlslwi. 2, 3, 5, 4

# Move to/from special purpose register mnemonics

# CHECK: mtspr 1, 2                      # encoding: [0x7c,0x41,0x03,0xa6]
         mtxer 2
# CHECK: mfspr 2, 1                      # encoding: [0x7c,0x41,0x02,0xa6]
         mfxer 2
# CHECK: mtlr 2                          # encoding: [0x7c,0x48,0x03,0xa6]
         mtlr 2
# CHECK: mflr 2                          # encoding: [0x7c,0x48,0x02,0xa6]
         mflr 2
# CHECK: mtctr 2                         # encoding: [0x7c,0x49,0x03,0xa6]
         mtctr 2
# CHECK: mfctr 2                         # encoding: [0x7c,0x49,0x02,0xa6]
         mfctr 2

# Miscellaneous mnemonics

# CHECK: nop                             # encoding: [0x60,0x00,0x00,0x00]
         nop
# CHECK: xori 0, 0, 0                    # encoding: [0x68,0x00,0x00,0x00]
         xnop
# CHECK: li 2, 128                       # encoding: [0x38,0x40,0x00,0x80]
         li 2, 128
# CHECK: lis 2, 128                      # encoding: [0x3c,0x40,0x00,0x80]
         lis 2, 128
# CHECK: la 2, 128(4)
         la 2, 128(4)
# CHECK: mr 2, 3                         # encoding: [0x7c,0x62,0x1b,0x78]
         mr 2, 3
# CHECK: or. 2, 3, 3                     # encoding: [0x7c,0x62,0x1b,0x79]
         mr. 2, 3
# CHECK: nor 2, 3, 3                     # encoding: [0x7c,0x62,0x18,0xf8]
         not 2, 3
# CHECK: nor. 2, 3, 3                    # encoding: [0x7c,0x62,0x18,0xf9]
         not. 2, 3
# CHECK: mtcrf 255, 2                    # encoding: [0x7c,0x4f,0xf1,0x20]
         mtcr 2

