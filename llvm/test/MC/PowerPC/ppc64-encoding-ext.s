
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

# FIXME: bt 2, target
# FIXME: bt target
# FIXME: bta 2, target
# FIXME: bta target
# FIXME: btlr 2
# FIXME: btlr
# FIXME: btctr 2
# FIXME: btctr
# FIXME: btl 2, target
# FIXME: btl target
# FIXME: btla 2, target
# FIXME: btla target
# FIXME: btlrl 2
# FIXME: btlrl
# FIXME: btctrl 2
# FIXME: btctrl

# FIXME: bf 2, target
# FIXME: bf target
# FIXME: bfa 2, target
# FIXME: bfa target
# FIXME: bflr 2
# FIXME: bflr
# FIXME: bfctr 2
# FIXME: bfctr
# FIXME: bfl 2, target
# FIXME: bfl target
# FIXME: bfla 2, target
# FIXME: bfla target
# FIXME: bflrl 2
# FIXME: bflrl
# FIXME: bfctrl 2
# FIXME: bfctrl

# CHECK: bdnz target                     # encoding: [0x42,0x00,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnz target
# FIXME: bdnza target
# CHECK: bdnzlr                          # encoding: [0x4e,0x00,0x00,0x20]
         bdnzlr
# CHECK: bdnzl target                    # encoding: [0x42,0x00,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnzl target
# FIXME: bdnzla target
# CHECK: bdnzlrl                         # encoding: [0x4e,0x00,0x00,0x21]
         bdnzlrl

# FIXME: bdnzt 2, target
# FIXME: bdnzt target
# FIXME: bdnzta 2, target
# FIXME: bdnzta target
# FIXME: bdnztlr 2
# FIXME: bdnztlr
# FIXME: bdnztl 2, target
# FIXME: bdnztl target
# FIXME: bdnztla 2, target
# FIXME: bdnztla target
# FIXME: bdnztlrl 2
# FIXME: bdnztlrl
# FIXME: bdnzf 2, target
# FIXME: bdnzf target
# FIXME: bdnzfa 2, target
# FIXME: bdnzfa target
# FIXME: bdnzflr 2
# FIXME: bdnzflr
# FIXME: bdnzfl 2, target
# FIXME: bdnzfl target
# FIXME: bdnzfla 2, target
# FIXME: bdnzfla target
# FIXME: bdnzflrl 2
# FIXME: bdnzflrl

# CHECK: bdz target                      # encoding: [0x42,0x40,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdz target
# FIXME: bdza target
# CHECK: bdzlr                           # encoding: [0x4e,0x40,0x00,0x20]
         bdzlr
# CHECK: bdzl target                     # encoding: [0x42,0x40,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdzl target
# FIXME: bdzla target
# CHECK: bdzlrl                          # encoding: [0x4e,0x40,0x00,0x21]
         bdzlrl

# FIXME: bdzt 2, target
# FIXME: bdzt target
# FIXME: bdzta 2, target
# FIXME: bdzta target
# FIXME: bdztlr 2
# FIXME: bdztlr
# FIXME: bdztl 2, target
# FIXME: bdztl target
# FIXME: bdztla 2, target
# FIXME: bdztla target
# FIXME: bdztlrl 2
# FIXME: bdztlrl
# FIXME: bdzf 2, target
# FIXME: bdzf target
# FIXME: bdzfa 2, target
# FIXME: bdzfa target
# FIXME: bdzflr 2
# FIXME: bdzflr
# FIXME: bdzfl 2, target
# FIXME: bdzfl target
# FIXME: bdzfla 2, target
# FIXME: bdzfla target
# FIXME: bdzflrl 2
# FIXME: bdzflrl

# CHECK: blt 2, target                   # encoding: [0x41,0x88,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blt 2, target
# CHECK: blt 0, target                   # encoding: [0x41,0x80,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blt target
# FIXME: blta 2, target
# FIXME: blta target
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
# FIXME: bltla 2, target
# FIXME: bltla target
# CHECK: bltlrl 2                        # encoding: [0x4d,0x88,0x00,0x21]
         bltlrl 2
# CHECK: bltlrl 0                        # encoding: [0x4d,0x80,0x00,0x21]
         bltlrl
# CHECK: bltctrl 2                       # encoding: [0x4d,0x88,0x04,0x21]
         bltctrl 2
# CHECK: bltctrl 0                       # encoding: [0x4d,0x80,0x04,0x21]
         bltctrl

# CHECK: ble 2, target                   # encoding: [0x40,0x89,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         ble 2, target
# CHECK: ble 0, target                   # encoding: [0x40,0x81,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         ble target
# FIXME: blea 2, target
# FIXME: blea target
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
# FIXME: blela 2, target
# FIXME: blela target
# CHECK: blelrl 2                        # encoding: [0x4c,0x89,0x00,0x21]
         blelrl 2
# CHECK: blelrl 0                        # encoding: [0x4c,0x81,0x00,0x21]
         blelrl
# CHECK: blectrl 2                       # encoding: [0x4c,0x89,0x04,0x21]
         blectrl 2
# CHECK: blectrl 0                       # encoding: [0x4c,0x81,0x04,0x21]
         blectrl

# CHECK: beq 2, target                   # encoding: [0x41,0x8a,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beq 2, target
# CHECK: beq 0, target                   # encoding: [0x41,0x82,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beq target
# FIXME: beqa 2, target
# FIXME: beqa target
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
# FIXME: beqla 2, target
# FIXME: beqla target
# CHECK: beqlrl 2                        # encoding: [0x4d,0x8a,0x00,0x21]
         beqlrl 2
# CHECK: beqlrl 0                        # encoding: [0x4d,0x82,0x00,0x21]
         beqlrl
# CHECK: beqctrl 2                       # encoding: [0x4d,0x8a,0x04,0x21]
         beqctrl 2
# CHECK: beqctrl 0                       # encoding: [0x4d,0x82,0x04,0x21]
         beqctrl

# CHECK: bge 2, target                   # encoding: [0x40,0x88,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bge 2, target
# CHECK: bge 0, target                   # encoding: [0x40,0x80,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bge target
# FIXME: bgea 2, target
# FIXME: bgea target
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
# FIXME: bgela 2, target
# FIXME: bgela target
# CHECK: bgelrl 2                        # encoding: [0x4c,0x88,0x00,0x21]
         bgelrl 2
# CHECK: bgelrl 0                        # encoding: [0x4c,0x80,0x00,0x21]
         bgelrl
# CHECK: bgectrl 2                       # encoding: [0x4c,0x88,0x04,0x21]
         bgectrl 2
# CHECK: bgectrl 0                       # encoding: [0x4c,0x80,0x04,0x21]
         bgectrl

# CHECK: bgt 2, target                   # encoding: [0x41,0x89,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgt 2, target
# CHECK: bgt 0, target                   # encoding: [0x41,0x81,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgt target
# FIXME: bgta 2, target
# FIXME: bgta target
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
# FIXME: bgtla 2, target
# FIXME: bgtla target
# CHECK: bgtlrl 2                        # encoding: [0x4d,0x89,0x00,0x21]
         bgtlrl 2
# CHECK: bgtlrl 0                        # encoding: [0x4d,0x81,0x00,0x21]
         bgtlrl
# CHECK: bgtctrl 2                       # encoding: [0x4d,0x89,0x04,0x21]
         bgtctrl 2
# CHECK: bgtctrl 0                       # encoding: [0x4d,0x81,0x04,0x21]
         bgtctrl

# CHECK: bge 2, target                   # encoding: [0x40,0x88,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnl 2, target
# CHECK: bge 0, target                   # encoding: [0x40,0x80,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnl target
# FIXME: bnla 2, target
# FIXME: bnla target
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
# FIXME: bnlla 2, target
# FIXME: bnlla target
# CHECK: bgelrl 2                        # encoding: [0x4c,0x88,0x00,0x21]
         bnllrl 2
# CHECK: bgelrl 0                        # encoding: [0x4c,0x80,0x00,0x21]
         bnllrl
# CHECK: bgectrl 2                       # encoding: [0x4c,0x88,0x04,0x21]
         bnlctrl 2
# CHECK: bgectrl 0                       # encoding: [0x4c,0x80,0x04,0x21]
         bnlctrl

# CHECK: bne 2, target                   # encoding: [0x40,0x8a,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bne 2, target
# CHECK: bne 0, target                   # encoding: [0x40,0x82,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bne target
# FIXME: bnea 2, target
# FIXME: bnea target
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
# FIXME: bnela 2, target
# FIXME: bnela target
# CHECK: bnelrl 2                        # encoding: [0x4c,0x8a,0x00,0x21]
         bnelrl 2
# CHECK: bnelrl 0                        # encoding: [0x4c,0x82,0x00,0x21]
         bnelrl
# CHECK: bnectrl 2                       # encoding: [0x4c,0x8a,0x04,0x21]
         bnectrl 2
# CHECK: bnectrl 0                       # encoding: [0x4c,0x82,0x04,0x21]
         bnectrl

# CHECK: ble 2, target                   # encoding: [0x40,0x89,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bng 2, target
# CHECK: ble 0, target                   # encoding: [0x40,0x81,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bng target
# FIXME: bnga 2, target
# FIXME: bnga target
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
# FIXME: bngla 2, target
# FIXME: bngla target
# CHECK: blelrl 2                        # encoding: [0x4c,0x89,0x00,0x21]
         bnglrl 2
# CHECK: blelrl 0                        # encoding: [0x4c,0x81,0x00,0x21]
         bnglrl
# CHECK: blectrl 2                       # encoding: [0x4c,0x89,0x04,0x21]
         bngctrl 2
# CHECK: blectrl 0                       # encoding: [0x4c,0x81,0x04,0x21]
         bngctrl

# CHECK: bun 2, target                   # encoding: [0x41,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bso 2, target
# CHECK: bun 0, target                   # encoding: [0x41,0x83,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bso target
# FIXME: bsoa 2, target
# FIXME: bsoa target
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
# FIXME: bsola 2, target
# FIXME: bsola target
# CHECK: bunlrl 2                        # encoding: [0x4d,0x8b,0x00,0x21]
         bsolrl 2
# CHECK: bunlrl 0                        # encoding: [0x4d,0x83,0x00,0x21]
         bsolrl
# CHECK: bunctrl 2                       # encoding: [0x4d,0x8b,0x04,0x21]
         bsoctrl 2
# CHECK: bunctrl 0                       # encoding: [0x4d,0x83,0x04,0x21]
         bsoctrl

# CHECK: bnu 2, target                   # encoding: [0x40,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bns 2, target
# CHECK: bnu 0, target                   # encoding: [0x40,0x83,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bns target
# FIXME: bnsa 2, target
# FIXME: bnsa target
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
# FIXME: bnsla 2, target
# FIXME: bnsla target
# CHECK: bnulrl 2                        # encoding: [0x4c,0x8b,0x00,0x21]
         bnslrl 2
# CHECK: bnulrl 0                        # encoding: [0x4c,0x83,0x00,0x21]
         bnslrl
# CHECK: bnuctrl 2                       # encoding: [0x4c,0x8b,0x04,0x21]
         bnsctrl 2
# CHECK: bnuctrl 0                       # encoding: [0x4c,0x83,0x04,0x21]
         bnsctrl

# CHECK: bun 2, target                   # encoding: [0x41,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bun 2, target
# CHECK: bun 0, target                   # encoding: [0x41,0x83,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bun target
# FIXME: buna 2, target
# FIXME: buna target
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
# FIXME: bunla 2, target
# FIXME: bunla target
# CHECK: bunlrl 2                        # encoding: [0x4d,0x8b,0x00,0x21]
         bunlrl 2
# CHECK: bunlrl 0                        # encoding: [0x4d,0x83,0x00,0x21]
         bunlrl
# CHECK: bunctrl 2                       # encoding: [0x4d,0x8b,0x04,0x21]
         bunctrl 2
# CHECK: bunctrl 0                       # encoding: [0x4d,0x83,0x04,0x21]
         bunctrl

# CHECK: bnu 2, target                   # encoding: [0x40,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnu 2, target
# CHECK: bnu 0, target                   # encoding: [0x40,0x83,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnu target
# FIXME: bnua 2, target
# FIXME: bnua target
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
# FIXME: bnula 2, target
# FIXME: bnula target
# CHECK: bnulrl 2                        # encoding: [0x4c,0x8b,0x00,0x21]
         bnulrl 2
# CHECK: bnulrl 0                        # encoding: [0x4c,0x83,0x00,0x21]
         bnulrl
# CHECK: bnuctrl 2                       # encoding: [0x4c,0x8b,0x04,0x21]
         bnuctrl 2
# CHECK: bnuctrl 0                       # encoding: [0x4c,0x83,0x04,0x21]
         bnuctrl

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
# FIXME: extrdi 2, 3, 4, 5
# FIXME: insrdi 2, 3, 4, 5
# FIXME: rotldi 2, 3, 4
# FIXME: rotrdi 2, 3, 4
# FIXME: rotld 2, 3, 4
# CHECK: sldi 2, 3, 4                    # encoding: [0x78,0x62,0x26,0xe4]
         sldi 2, 3, 4
# CHECK: rldicl 2, 3, 60, 4              # encoding: [0x78,0x62,0xe1,0x02]
         srdi 2, 3, 4
# FIXME: clrldi 2, 3, 4
# FIXME: clrrdi 2, 3, 4
# FIXME: clrlsldi 2, 3, 4, 5

# FIXME: extlwi 2, 3, 4, 5
# FIXME: extrwi 2, 3, 4, 5
# FIXME: inslwi 2, 3, 4, 5
# FIXME: insrwi 2, 3, 4, 5
# FIXME: rotlwi 2, 3, 4
# FIXME: rotrwi 2, 3, 4
# FIXME: rotlw 2, 3, 4
# CHECK: slwi 2, 3, 4                    # encoding: [0x54,0x62,0x20,0x36]
         slwi 2, 3, 4
# CHECK: srwi 2, 3, 4                    # encoding: [0x54,0x62,0xe1,0x3e]
         srwi 2, 3, 4
# FIXME: clrlwi 2, 3, 4
# FIXME: clrrwi 2, 3, 4
# FIXME: clrlslwi 2, 3, 4, 5

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
# FIXME: not 2, 3

