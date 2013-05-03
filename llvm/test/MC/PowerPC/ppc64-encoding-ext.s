
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# FIXME: Condition register bit symbols

# Branch mnemonics

# CHECK: blr                             # encoding: [0x4e,0x80,0x00,0x20]
         blr
# CHECK: bctr                            # encoding: [0x4e,0x80,0x04,0x20]
         bctr
# FIXME: blrl
# CHECK: bctrl                           # encoding: [0x4e,0x80,0x04,0x21]
         bctrl

# FIXME: bt 2, target
# FIXME: bta 2, target
# FIXME: btlr 2
# FIXME: btctr 2
# FIXME: btl 2, target
# FIXME: btla 2, target
# FIXME: btlrl 2
# FIXME: btctrl 2

# FIXME: bf 2, target
# FIXME: bfa 2, target
# FIXME: bflr 2
# FIXME: bfctr 2
# FIXME: bfl 2, target
# FIXME: bfla 2, target
# FIXME: bflrl 2
# FIXME: bfctrl 2

# CHECK: bdnz target                     # encoding: [0x42,0x00,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdnz target
# FIXME: bdnza target
# CHECK: bdnzlr                          # encoding: [0x4e,0x00,0x00,0x20]
         bdnzlr
# FIXME: bdnzl target
# FIXME: bdnzla target
# FIXME: bdnzlrl

# FIXME: bdnzt 2, target
# FIXME: bdnzta 2, target
# FIXME: bdnztlr 2
# FIXME: bdnztl 2, target
# FIXME: bdnztla 2, target
# FIXME: bdnztlrl 2
# FIXME: bdnzf 2, target
# FIXME: bdnzfa 2, target
# FIXME: bdnzflr 2
# FIXME: bdnzfl 2, target
# FIXME: bdnzfla 2, target
# FIXME: bdnzflrl 2

# CHECK: bdz target                      # encoding: [0x42,0x40,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bdz target
# FIXME: bdza target
# CHECK: bdzlr                           # encoding: [0x4e,0x40,0x00,0x20]
         bdzlr
# FIXME: bdzl target
# FIXME: bdzla target

# FIXME: bdzlrl
# FIXME: bdzt 2, target
# FIXME: bdzta 2, target
# FIXME: bdztlr 2
# FIXME: bdztl 2, target
# FIXME: bdztla 2, target
# FIXME: bdztlrl 2
# FIXME: bdzf 2, target
# FIXME: bdzfa 2, target
# FIXME: bdzflr 2
# FIXME: bdzfl 2, target
# FIXME: bdzfla 2, target
# FIXME: bdzflrl 2

# CHECK: blt 2, target                   # encoding: [0x41,0x88,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         blt 2, target
# FIXME: blta 2, target
# CHECK: bltlr 2                         # encoding: [0x4d,0x88,0x00,0x20]
         bltlr 2
# CHECK: bltctr 2                        # encoding: [0x4d,0x88,0x04,0x20]
         bltctr 2
# FIXME: bltl 2, target
# FIXME: bltla 2, target
# FIXME: bltlrl 2
# CHECK: bltctrl 2                       # encoding: [0x4d,0x88,0x04,0x21]
         bltctrl 2

# CHECK: ble 2, target                   # encoding: [0x40,0x89,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         ble 2, target
# FIXME: blea 2, target
# CHECK: blelr 2                         # encoding: [0x4c,0x89,0x00,0x20]
         blelr 2
# CHECK: blectr 2                        # encoding: [0x4c,0x89,0x04,0x20]
         blectr 2
# FIXME: blel 2, target
# FIXME: blela 2, target
# FIXME: blelrl 2
# CHECK: blectrl 2                       # encoding: [0x4c,0x89,0x04,0x21]
         blectrl 2

# CHECK: beq 2, target                   # encoding: [0x41,0x8a,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         beq 2, target
# FIXME: beqa 2, target
# CHECK: beqlr 2                         # encoding: [0x4d,0x8a,0x00,0x20]
         beqlr 2
# CHECK: beqctr 2                        # encoding: [0x4d,0x8a,0x04,0x20]
         beqctr 2
# FIXME: beql 2, target
# FIXME: beqla 2, target
# FIXME: beqlrl 2
# CHECK: beqctrl 2                       # encoding: [0x4d,0x8a,0x04,0x21]
         beqctrl 2

# CHECK: bge 2, target                   # encoding: [0x40,0x88,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bge 2, target
# FIXME: bgea 2, target
# CHECK: bgelr 2                         # encoding: [0x4c,0x88,0x00,0x20]
         bgelr 2
# CHECK: bgectr 2                        # encoding: [0x4c,0x88,0x04,0x20]
         bgectr 2
# FIXME: bgel 2, target
# FIXME: bgela 2, target
# FIXME: bgelrl 2
# CHECK: bgectrl 2                       # encoding: [0x4c,0x88,0x04,0x21]
         bgectrl 2

# CHECK: bgt 2, target                   # encoding: [0x41,0x89,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bgt 2, target
# FIXME: bgta 2, target
# CHECK: bgtlr 2                         # encoding: [0x4d,0x89,0x00,0x20]
         bgtlr 2
# CHECK: bgtctr 2                        # encoding: [0x4d,0x89,0x04,0x20]
         bgtctr 2
# FIXME: bgtl 2, target
# FIXME: bgtla 2, target
# FIXME: bgtlrl 2
# CHECK: bgtctrl 2                       # encoding: [0x4d,0x89,0x04,0x21]
         bgtctrl 2

# CHECK: bge 2, target                   # encoding: [0x40,0x88,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnl 2, target
# FIXME: bnla 2, target
# CHECK: bgelr 2                         # encoding: [0x4c,0x88,0x00,0x20]
         bnllr 2
# CHECK: bgectr 2                        # encoding: [0x4c,0x88,0x04,0x20]
         bnlctr 2
# FIXME: bnll 2, target
# FIXME: bnlla 2, target
# FIXME: bnllrl 2
# CHECK: bgectrl 2                       # encoding: [0x4c,0x88,0x04,0x21]
         bnlctrl 2

# CHECK: bne 2, target                   # encoding: [0x40,0x8a,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bne 2, target
# FIXME: bnea 2, target
# CHECK: bnelr 2                         # encoding: [0x4c,0x8a,0x00,0x20]
         bnelr 2
# CHECK: bnectr 2                        # encoding: [0x4c,0x8a,0x04,0x20]
         bnectr 2
# FIXME: bnel 2, target
# FIXME: bnela 2, target
# FIXME: bnelrl 2
# CHECK: bnectrl 2                       # encoding: [0x4c,0x8a,0x04,0x21]
         bnectrl 2

# CHECK: ble 2, target                   # encoding: [0x40,0x89,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bng 2, target
# FIXME: bnga 2, target
# CHECK: blelr 2                         # encoding: [0x4c,0x89,0x00,0x20]
         bnglr 2
# CHECK: blectr 2                        # encoding: [0x4c,0x89,0x04,0x20]
         bngctr 2
# FIXME: bngl 2, target
# FIXME: bngla 2, target
# FIXME: bnglrl 2
# CHECK: blectrl 2                       # encoding: [0x4c,0x89,0x04,0x21]
         bngctrl 2

# CHECK: bun 2, target                   # encoding: [0x41,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bso 2, target
# FIXME: bsoa 2, target
# CHECK: bunlr 2                         # encoding: [0x4d,0x8b,0x00,0x20]
         bsolr 2
# CHECK: bunctr 2                        # encoding: [0x4d,0x8b,0x04,0x20]
         bsoctr 2
# FIXME: bsol 2, target
# FIXME: bsola 2, target
# FIXME: bsolrl 2
# CHECK: bunctrl 2                       # encoding: [0x4d,0x8b,0x04,0x21]
         bsoctrl 2

# CHECK: bnu 2, target                   # encoding: [0x40,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bns 2, target
# FIXME: bnsa 2, target
# CHECK: bnulr 2                         # encoding: [0x4c,0x8b,0x00,0x20]
         bnslr 2
# CHECK: bnuctr 2                        # encoding: [0x4c,0x8b,0x04,0x20]
         bnsctr 2
# FIXME: bnsl 2, target
# FIXME: bnsla 2, target
# FIXME: bnslrl 2
# CHECK: bnuctrl 2                       # encoding: [0x4c,0x8b,0x04,0x21]
         bnsctrl 2

# CHECK: bun 2, target                   # encoding: [0x41,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bun 2, target
# FIXME: buna 2, target
# CHECK: bunlr 2                         # encoding: [0x4d,0x8b,0x00,0x20]
         bunlr 2
# CHECK: bunctr 2                        # encoding: [0x4d,0x8b,0x04,0x20]
         bunctr 2
# FIXME: bunl 2, target
# FIXME: bunla 2, target
# FIXME: bunlrl 2
# CHECK: bunctrl 2                       # encoding: [0x4d,0x8b,0x04,0x21]
         bunctrl 2

# CHECK: bnu 2, target                   # encoding: [0x40,0x8b,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bnu 2, target
# FIXME: bnua 2, target
# CHECK: bnulr 2                         # encoding: [0x4c,0x8b,0x00,0x20]
         bnulr 2
# CHECK: bnuctr 2                        # encoding: [0x4c,0x8b,0x04,0x20]
         bnuctr 2
# FIXME: bnul 2, target
# FIXME: bnula 2, target
# FIXME: bnulrl 2
# CHECK: bnuctrl 2                       # encoding: [0x4c,0x8b,0x04,0x21]
         bnuctrl 2

# FIXME: Condition register logical mnemonics

# FIXME: Subtract mnemonics

# Compare mnemonics

# CHECK: cmpdi 2, 3, 128                 # encoding: [0x2d,0x23,0x00,0x80]
         cmpdi 2, 3, 128
# CHECK: cmpd 2, 3, 4                    # encoding: [0x7d,0x23,0x20,0x00]
         cmpd 2, 3, 4
# CHECK: cmpldi 2, 3, 128                # encoding: [0x29,0x23,0x00,0x80]
         cmpldi 2, 3, 128
# CHECK: cmpld 2, 3, 4                   # encoding: [0x7d,0x23,0x20,0x40]
         cmpld 2, 3, 4

# CHECK: cmpwi 2, 3, 128                 # encoding: [0x2d,0x03,0x00,0x80]
         cmpwi 2, 3, 128
# CHECK: cmpw 2, 3, 4                    # encoding: [0x7d,0x03,0x20,0x00]
         cmpw 2, 3, 4
# CHECK: cmplwi 2, 3, 128                # encoding: [0x29,0x03,0x00,0x80]
         cmplwi 2, 3, 128
# CHECK: cmplw 2, 3, 4                   # encoding: [0x7d,0x03,0x20,0x40]
         cmplw 2, 3, 4

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

