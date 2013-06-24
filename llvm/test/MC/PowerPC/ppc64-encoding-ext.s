
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# FIXME: Condition register bit symbols

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

# FIXME: Condition register logical mnemonics

# FIXME: Subtract mnemonics

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

# FIXME: Trap mnemonics

# Rotate and shift mnemonics

# FIXME: extldi 2, 3, 4, 5
# FIXME: extldi. 2, 3, 4, 5
# FIXME: extrdi 2, 3, 4, 5
# FIXME: extrdi. 2, 3, 4, 5
# FIXME: insrdi 2, 3, 4, 5
# FIXME: insrdi. 2, 3, 4, 5
# FIXME: rotldi 2, 3, 4
# FIXME: rotldi. 2, 3, 4
# FIXME: rotrdi 2, 3, 4
# FIXME: rotrdi. 2, 3, 4
# FIXME: rotld 2, 3, 4
# FIXME: rotld. 2, 3, 4
# CHECK: sldi 2, 3, 4                    # encoding: [0x78,0x62,0x26,0xe4]
         sldi 2, 3, 4
# FIXME: sldi. 2, 3, 4
# CHECK: rldicl 2, 3, 60, 4              # encoding: [0x78,0x62,0xe1,0x02]
         srdi 2, 3, 4
# FIXME: srdi. 2, 3, 4
# FIXME: clrldi 2, 3, 4
# FIXME: clrldi. 2, 3, 4
# FIXME: clrrdi 2, 3, 4
# FIXME: clrrdi. 2, 3, 4
# FIXME: clrlsldi 2, 3, 4, 5
# FIXME: clrlsldi. 2, 3, 4, 5

# FIXME: extlwi 2, 3, 4, 5
# FIXME: extlwi. 2, 3, 4, 5
# FIXME: extrwi 2, 3, 4, 5
# FIXME: extrwi. 2, 3, 4, 5
# FIXME: inslwi 2, 3, 4, 5
# FIXME: inslwi. 2, 3, 4, 5
# FIXME: insrwi 2, 3, 4, 5
# FIXME: insrwi. 2, 3, 4, 5
# FIXME: rotlwi 2, 3, 4
# FIXME: rotlwi. 2, 3, 4
# FIXME: rotrwi 2, 3, 4
# FIXME: rotrwi. 2, 3, 4
# FIXME: rotlw 2, 3, 4
# FIXME: rotlw. 2, 3, 4
# CHECK: slwi 2, 3, 4                    # encoding: [0x54,0x62,0x20,0x36]
         slwi 2, 3, 4
# FIXME: slwi. 2, 3, 4
# CHECK: srwi 2, 3, 4                    # encoding: [0x54,0x62,0xe1,0x3e]
         srwi 2, 3, 4
# FIXME: srwi. 2, 3, 4
# FIXME: clrlwi 2, 3, 4
# FIXME: clrlwi. 2, 3, 4
# FIXME: clrrwi 2, 3, 4
# FIXME: clrrwi. 2, 3, 4
# FIXME: clrlslwi 2, 3, 4, 5
# FIXME: clrlslwi. 2, 3, 4, 5

# Move to/from special purpose register mnemonics

# FIXME: mtxer 2
# FIXME: mfxer 2
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
# FIXME: xnop
# CHECK: li 2, 128                       # encoding: [0x38,0x40,0x00,0x80]
         li 2, 128
# CHECK: lis 2, 128                      # encoding: [0x3c,0x40,0x00,0x80]
         lis 2, 128
# FIXME: la 2, 128(4)
# CHECK: mr 2, 3                         # encoding: [0x7c,0x62,0x1b,0x78]
         mr 2, 3
# FIXME: mr. 2, 3
# FIXME: not 2, 3
# FIXME: not. 2, 3

