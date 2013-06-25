
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# Branch facility

# Branch instructions

# CHECK: b target                        # encoding: [0b010010AA,A,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24
         b target
# CHECK: ba target                       # encoding: [0b010010AA,A,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24abs
         ba target
# CHECK: bl target                       # encoding: [0b010010AA,A,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24
         bl target
# CHECK: bla target                      # encoding: [0b010010AA,A,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24abs
         bla target

# CHECK: bc 4, 10, target                # encoding: [0x40,0x8a,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bc 4, 10, target
# CHECK: bca 4, 10, target               # encoding: [0x40,0x8a,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bca 4, 10, target
# CHECK: bcl 4, 10, target               # encoding: [0x40,0x8a,A,0bAAAAAA01]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
         bcl 4, 10, target
# CHECK: bcla 4, 10, target              # encoding: [0x40,0x8a,A,0bAAAAAA11]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
         bcla 4, 10, target

# CHECK: bclr 4, 10, 3                   # encoding: [0x4c,0x8a,0x18,0x20]
         bclr 4, 10, 3
# CHECK: bclr 4, 10, 0                   # encoding: [0x4c,0x8a,0x00,0x20]
         bclr 4, 10
# CHECK: bclrl 4, 10, 3                  # encoding: [0x4c,0x8a,0x18,0x21]
         bclrl 4, 10, 3
# CHECK: bclrl 4, 10, 0                  # encoding: [0x4c,0x8a,0x00,0x21]
         bclrl 4, 10
# CHECK: bcctr 4, 10, 3                  # encoding: [0x4c,0x8a,0x1c,0x20]
         bcctr 4, 10, 3
# CHECK: bcctr 4, 10, 0                  # encoding: [0x4c,0x8a,0x04,0x20]
         bcctr 4, 10
# CHECK: bcctrl 4, 10, 3                 # encoding: [0x4c,0x8a,0x1c,0x21]
         bcctrl 4, 10, 3
# CHECK: bcctrl 4, 10, 0                 # encoding: [0x4c,0x8a,0x04,0x21]
         bcctrl 4, 10

# Condition register instructions

# FIXME: crand 2, 3, 4
# FIXME: crnand 2, 3, 4
# CHECK: cror 2, 3, 4                    # encoding: [0x4c,0x43,0x23,0x82]
         cror 2, 3, 4
# FIXME: crxor 2, 3, 4
# FIXME: crnor 2, 3, 4
# CHECK: creqv 2, 3, 4                   # encoding: [0x4c,0x43,0x22,0x42]
         creqv 2, 3, 4
# FIXME: crandc 2, 3, 4
# FIXME: crorc 2, 3, 4
# CHECK: mcrf 2, 3                       # encoding: [0x4d,0x0c,0x00,0x00]
         mcrf 2, 3

# System call instruction

# CHECK: sc 1                            # encoding: [0x44,0x00,0x00,0x22]
         sc 1
# CHECK: sc 0                            # encoding: [0x44,0x00,0x00,0x02]
         sc

# Fixed-point facility

# Fixed-point load instructions

# CHECK: lbz 2, 128(4)                   # encoding: [0x88,0x44,0x00,0x80]
         lbz 2, 128(4)
# CHECK: lbzx 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0xae]
         lbzx 2, 3, 4
# CHECK: lbzu 2, 128(4)                  # encoding: [0x8c,0x44,0x00,0x80]
         lbzu 2, 128(4)
# CHECK: lbzux 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0xee]
         lbzux 2, 3, 4
# CHECK: lhz 2, 128(4)                   # encoding: [0xa0,0x44,0x00,0x80]
         lhz 2, 128(4)
# CHECK: lhzx 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0x2e]
         lhzx 2, 3, 4
# CHECK: lhzu 2, 128(4)                  # encoding: [0xa4,0x44,0x00,0x80]
         lhzu 2, 128(4)
# CHECK: lhzux 2, 3, 4                   # encoding: [0x7c,0x43,0x22,0x6e]
         lhzux 2, 3, 4
# CHECK: lha 2, 128(4)                   # encoding: [0xa8,0x44,0x00,0x80]
         lha 2, 128(4)
# CHECK: lhax 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0xae]
         lhax 2, 3, 4
# CHECK: lhau 2, 128(4)                  # encoding: [0xac,0x44,0x00,0x80]
         lhau 2, 128(4)
# CHECK: lhaux 2, 3, 4                   # encoding: [0x7c,0x43,0x22,0xee]
         lhaux 2, 3, 4
# CHECK: lwz 2, 128(4)                   # encoding: [0x80,0x44,0x00,0x80]
         lwz 2, 128(4)
# CHECK: lwzx 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x2e]
         lwzx 2, 3, 4
# CHECK: lwzu 2, 128(4)                  # encoding: [0x84,0x44,0x00,0x80]
         lwzu 2, 128(4)
# CHECK: lwzux 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x6e]
         lwzux 2, 3, 4
# CHECK: lwa 2, 128(4)                   # encoding: [0xe8,0x44,0x00,0x82]
         lwa 2, 128(4)
# CHECK: lwax 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0xaa]
         lwax 2, 3, 4
# CHECK: lwaux 2, 3, 4                   # encoding: [0x7c,0x43,0x22,0xea]
         lwaux 2, 3, 4
# CHECK: ld 2, 128(4)                    # encoding: [0xe8,0x44,0x00,0x80]
         ld 2, 128(4)
# CHECK: ldx 2, 3, 4                     # encoding: [0x7c,0x43,0x20,0x2a]
         ldx 2, 3, 4
# CHECK: ldu 2, 128(4)                   # encoding: [0xe8,0x44,0x00,0x81]
         ldu 2, 128(4)
# CHECK: ldux 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x6a]
         ldux 2, 3, 4

# Fixed-point store instructions

# CHECK: stb 2, 128(4)                   # encoding: [0x98,0x44,0x00,0x80]
         stb 2, 128(4)
# CHECK: stbx 2, 3, 4                    # encoding: [0x7c,0x43,0x21,0xae]
         stbx 2, 3, 4
# CHECK: stbu 2, 128(4)                  # encoding: [0x9c,0x44,0x00,0x80]
         stbu 2, 128(4)
# CHECK: stbux 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0xee]
         stbux 2, 3, 4
# CHECK: sth 2, 128(4)                   # encoding: [0xb0,0x44,0x00,0x80]
         sth 2, 128(4)
# CHECK: sthx 2, 3, 4                    # encoding: [0x7c,0x43,0x23,0x2e]
         sthx 2, 3, 4
# CHECK: sthu 2, 128(4)                  # encoding: [0xb4,0x44,0x00,0x80]
         sthu 2, 128(4)
# CHECK: sthux 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0x6e]
         sthux 2, 3, 4
# CHECK: stw 2, 128(4)                   # encoding: [0x90,0x44,0x00,0x80]
         stw 2, 128(4)
# CHECK: stwx 2, 3, 4                    # encoding: [0x7c,0x43,0x21,0x2e]
         stwx 2, 3, 4
# CHECK: stwu 2, 128(4)                  # encoding: [0x94,0x44,0x00,0x80]
         stwu 2, 128(4)
# CHECK: stwux 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0x6e]
         stwux 2, 3, 4
# CHECK: std 2, 128(4)                   # encoding: [0xf8,0x44,0x00,0x80]
         std 2, 128(4)
# CHECK: stdx 2, 3, 4                    # encoding: [0x7c,0x43,0x21,0x2a]
         stdx 2, 3, 4
# CHECK: stdu 2, 128(4)                  # encoding: [0xf8,0x44,0x00,0x81]
         stdu 2, 128(4)
# CHECK: stdux 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0x6a]
         stdux 2, 3, 4

# Fixed-point load and store with byte reversal instructions

# CHECK: lhbrx 2, 3, 4                   # encoding: [0x7c,0x43,0x26,0x2c]
         lhbrx 2, 3, 4
# CHECK: sthbrx 2, 3, 4                  # encoding: [0x7c,0x43,0x27,0x2c]
         sthbrx 2, 3, 4
# CHECK: lwbrx 2, 3, 4                   # encoding: [0x7c,0x43,0x24,0x2c]
         lwbrx 2, 3, 4
# CHECK: stwbrx 2, 3, 4                  # encoding: [0x7c,0x43,0x25,0x2c]
         stwbrx 2, 3, 4
# CHECK: ldbrx 2, 3, 4                   # encoding: [0x7c,0x43,0x24,0x28]
         ldbrx 2, 3, 4
# CHECK: stdbrx 2, 3, 4                  # encoding: [0x7c,0x43,0x25,0x28]
         stdbrx 2, 3, 4

# FIXME: Fixed-point load and store multiple instructions

# FIXME: Fixed-point move assist instructions

# Fixed-point arithmetic instructions

# CHECK: addi 2, 3, 128                  # encoding: [0x38,0x43,0x00,0x80]
         addi 2, 3, 128
# CHECK: addis 2, 3, 128                 # encoding: [0x3c,0x43,0x00,0x80]
         addis 2, 3, 128
# CHECK: add 2, 3, 4                     # encoding: [0x7c,0x43,0x22,0x14]
         add 2, 3, 4
# CHECK: add. 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0x15]
         add. 2, 3, 4
# FIXME: addo 2, 3, 4
# FIXME: addo. 2, 3, 4
# CHECK: subf 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x50]
         subf 2, 3, 4
# CHECK: subf. 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x51]
         subf. 2, 3, 4
# FIXME: subfo 2, 3, 4
# FIXME: subfo. 2, 3, 4
# CHECK: addic 2, 3, 128                 # encoding: [0x30,0x43,0x00,0x80]
         addic 2, 3, 128
# CHECK: addic. 2, 3, 128                # encoding: [0x34,0x43,0x00,0x80]
         addic. 2, 3, 128
# CHECK: subfic 2, 3, 4                  # encoding: [0x20,0x43,0x00,0x04]
         subfic 2, 3, 4

# CHECK: addc 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x14]
         addc 2, 3, 4
# CHECK: addc. 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x15]
         addc. 2, 3, 4
# FIXME: addco 2, 3, 4
# FIXME: addco. 2, 3, 4
# CHECK: subfc 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x10]
         subfc 2, 3, 4
# CHECK: subfc 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x10]
         subfc 2, 3, 4
# FIXME: subfco 2, 3, 4
# FIXME: subfco. 2, 3, 4

# CHECK: adde 2, 3, 4                    # encoding: [0x7c,0x43,0x21,0x14]
         adde 2, 3, 4
# CHECK: adde. 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0x15]
         adde. 2, 3, 4
# FIXME: addeo 2, 3, 4
# FIXME: addeo. 2, 3, 4
# CHECK: subfe 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0x10]
         subfe 2, 3, 4
# CHECK: subfe. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x11]
         subfe. 2, 3, 4
# FIXME: subfeo 2, 3, 4
# FIXME: subfeo. 2, 3, 4

# CHECK: addme 2, 3                      # encoding: [0x7c,0x43,0x01,0xd4]
         addme 2, 3
# CHECK: addme. 2, 3                     # encoding: [0x7c,0x43,0x01,0xd5]
         addme. 2, 3
# FIXME: addmeo 2, 3
# FIXME: addmeo. 2, 3
# CHECK: subfme 2, 3                     # encoding: [0x7c,0x43,0x01,0xd0]
         subfme 2, 3
# CHECK: subfme. 2, 3                    # encoding: [0x7c,0x43,0x01,0xd1]
         subfme. 2, 3
# FIXME: subfmeo 2, 3
# FIXME: subfmeo. 2, 3

# CHECK: addze 2, 3                      # encoding: [0x7c,0x43,0x01,0x94]
         addze 2, 3
# CHECK: addze. 2, 3                     # encoding: [0x7c,0x43,0x01,0x95]
         addze. 2, 3
# FIXME: addzeo 2, 3
# FIXME: addzeo. 2, 3
# CHECK: subfze 2, 3                     # encoding: [0x7c,0x43,0x01,0x90]
         subfze 2, 3
# CHECK: subfze. 2, 3                    # encoding: [0x7c,0x43,0x01,0x91]
         subfze. 2, 3
# FIXME: subfzeo 2, 3
# FIXME: subfzeo. 2, 3

# CHECK: neg 2, 3                        # encoding: [0x7c,0x43,0x00,0xd0]
         neg 2, 3
# CHECK: neg. 2, 3                       # encoding: [0x7c,0x43,0x00,0xd1]
         neg. 2, 3
# FIXME: nego 2, 3
# FIXME: nego. 2, 3

# CHECK: mulli 2, 3, 128                 # encoding: [0x1c,0x43,0x00,0x80]
         mulli 2, 3, 128
# CHECK: mulhw 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x96]
         mulhw 2, 3, 4
# CHECK: mulhw. 2, 3, 4                  # encoding: [0x7c,0x43,0x20,0x97]
         mulhw. 2, 3, 4
# CHECK: mullw 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0xd6]
         mullw 2, 3, 4
# CHECK: mullw. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0xd7]
         mullw. 2, 3, 4
# FIXME: mullwo 2, 3, 4
# FIXME: mullwo. 2, 3, 4
# CHECK: mulhwu 2, 3, 4                  # encoding: [0x7c,0x43,0x20,0x16]
         mulhwu 2, 3, 4
# CHECK: mulhwu. 2, 3, 4                 # encoding: [0x7c,0x43,0x20,0x17]
         mulhwu. 2, 3, 4

# CHECK: divw 2, 3, 4                    # encoding: [0x7c,0x43,0x23,0xd6]
         divw 2, 3, 4
# CHECK: divw. 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0xd7]
         divw. 2, 3, 4
# FIXME: divwo 2, 3, 4
# FIXME: divwo. 2, 3, 4
# CHECK: divwu 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0x96]
         divwu 2, 3, 4
# CHECK: divwu. 2, 3, 4                  # encoding: [0x7c,0x43,0x23,0x97]
         divwu. 2, 3, 4
# FIXME: divwuo 2, 3, 4
# FIXME: divwuo. 2, 3, 4
# FIXME: divwe 2, 3, 4
# FIXME: divwe. 2, 3, 4
# FIXME: divweo 2, 3, 4
# FIXME: divweo. 2, 3, 4
# FIXME: divweu 2, 3, 4
# FIXME: divweu. 2, 3, 4
# FIXME: divweuo 2, 3, 4
# FIXME: divweuo. 2, 3, 4

# CHECK: mulld 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0xd2]
         mulld 2, 3, 4
# CHECK: mulld. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0xd3]
         mulld. 2, 3, 4
# FIXME: mulldo 2, 3, 4
# FIXME: mulldo. 2, 3, 4
# CHECK: mulhd 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x92]
         mulhd 2, 3, 4
# CHECK: mulhd. 2, 3, 4                  # encoding: [0x7c,0x43,0x20,0x93]
         mulhd. 2, 3, 4
# CHECK: mulhdu 2, 3, 4                  # encoding: [0x7c,0x43,0x20,0x12]
         mulhdu 2, 3, 4
# CHECK: mulhdu. 2, 3, 4                 # encoding: [0x7c,0x43,0x20,0x13]
         mulhdu. 2, 3, 4

# CHECK: divd 2, 3, 4                    # encoding: [0x7c,0x43,0x23,0xd2]
         divd 2, 3, 4
# CHECK: divd. 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0xd3]
         divd. 2, 3, 4
# FIXME: divdo 2, 3, 4
# FIXME: divdo. 2, 3, 4
# CHECK: divdu 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0x92]
         divdu 2, 3, 4
# CHECK: divdu. 2, 3, 4                  # encoding: [0x7c,0x43,0x23,0x93]
         divdu. 2, 3, 4
# FIXME: divduo 2, 3, 4
# FIXME: divduo. 2, 3, 4
# FIXME: divde 2, 3, 4
# FIXME: divde. 2, 3, 4
# FIXME: divdeo 2, 3, 4
# FIXME: divdeo. 2, 3, 4
# FIXME: divdeu 2, 3, 4
# FIXME: divdeu. 2, 3, 4
# FIXME: divdeuo 2, 3, 4
# FIXME: divdeuo. 2, 3, 4

# FIXME: Fixed-point compare instructions

# FIXME: Fixed-point trap instructions

# Fixed-point select

# CHECK: isel 2, 3, 4, 5                 # encoding: [0x7c,0x43,0x21,0x5e]
         isel 2, 3, 4, 5

# Fixed-point logical instructions

# CHECK: andi. 2, 3, 128                 # encoding: [0x70,0x62,0x00,0x80]
         andi. 2, 3, 128
# CHECK: andis. 2, 3, 128                # encoding: [0x74,0x62,0x00,0x80]
         andis. 2, 3, 128
# CHECK: ori 2, 3, 128                   # encoding: [0x60,0x62,0x00,0x80]
         ori 2, 3, 128
# CHECK: oris 2, 3, 128                  # encoding: [0x64,0x62,0x00,0x80]
         oris 2, 3, 128
# CHECK: xori 2, 3, 128                  # encoding: [0x68,0x62,0x00,0x80]
         xori 2, 3, 128
# CHECK: xoris 2, 3, 128                 # encoding: [0x6c,0x62,0x00,0x80]
         xoris 2, 3, 128
# CHECK: and 2, 3, 4                     # encoding: [0x7c,0x62,0x20,0x38]
         and 2, 3, 4
# CHECK: and. 2, 3, 4                    # encoding: [0x7c,0x62,0x20,0x39]
         and. 2, 3, 4
# CHECK: xor 2, 3, 4                     # encoding: [0x7c,0x62,0x22,0x78]
         xor 2, 3, 4
# CHECK: xor. 2, 3, 4                    # encoding: [0x7c,0x62,0x22,0x79]
         xor. 2, 3, 4
# CHECK: nand 2, 3, 4                    # encoding: [0x7c,0x62,0x23,0xb8]
         nand 2, 3, 4
# CHECK: nand. 2, 3, 4                   # encoding: [0x7c,0x62,0x23,0xb9]
         nand. 2, 3, 4
# CHECK: or 2, 3, 4                      # encoding: [0x7c,0x62,0x23,0x78]
         or 2, 3, 4
# CHECK: or. 2, 3, 4                     # encoding: [0x7c,0x62,0x23,0x79]
         or. 2, 3, 4
# CHECK: nor 2, 3, 4                     # encoding: [0x7c,0x62,0x20,0xf8]
         nor 2, 3, 4
# CHECK: nor. 2, 3, 4                    # encoding: [0x7c,0x62,0x20,0xf9]
         nor. 2, 3, 4
# CHECK: eqv 2, 3, 4                     # encoding: [0x7c,0x62,0x22,0x38]
         eqv 2, 3, 4
# CHECK: eqv. 2, 3, 4                    # encoding: [0x7c,0x62,0x22,0x39]
         eqv. 2, 3, 4
# CHECK: andc 2, 3, 4                    # encoding: [0x7c,0x62,0x20,0x78]
         andc 2, 3, 4
# CHECK: andc. 2, 3, 4                   # encoding: [0x7c,0x62,0x20,0x79]
         andc. 2, 3, 4
# CHECK: orc 2, 3, 4                     # encoding: [0x7c,0x62,0x23,0x38]
         orc 2, 3, 4
# CHECK: orc. 2, 3, 4                    # encoding: [0x7c,0x62,0x23,0x39]
         orc. 2, 3, 4

# CHECK: extsb 2, 3                      # encoding: [0x7c,0x62,0x07,0x74]
         extsb 2, 3
# CHECK: extsb. 2, 3                     # encoding: [0x7c,0x62,0x07,0x75]
         extsb. 2, 3
# CHECK: extsh 2, 3                      # encoding: [0x7c,0x62,0x07,0x34]
         extsh 2, 3
# CHECK: extsh. 2, 3                     # encoding: [0x7c,0x62,0x07,0x35]
         extsh. 2, 3

# CHECK: cntlzw 2, 3                     # encoding: [0x7c,0x62,0x00,0x34]
         cntlzw 2, 3
# CHECK: cntlzw. 2, 3                    # encoding: [0x7c,0x62,0x00,0x35]
         cntlzw. 2, 3
# FIXME: cmpb 2, 3, 4
# FIXME: popcntb 2, 3
# CHECK: popcntw 2, 3                    # encoding: [0x7c,0x62,0x02,0xf4]
         popcntw 2, 3
# FIXME: prtyd 2, 3
# FIXME: prtyw 2, 3

# CHECK: extsw 2, 3                      # encoding: [0x7c,0x62,0x07,0xb4]
         extsw 2, 3
# CHECK: extsw. 2, 3                     # encoding: [0x7c,0x62,0x07,0xb5]
         extsw. 2, 3

# CHECK: cntlzd 2, 3                     # encoding: [0x7c,0x62,0x00,0x74]
         cntlzd 2, 3
# CHECK: cntlzd. 2, 3                    # encoding: [0x7c,0x62,0x00,0x75]
         cntlzd. 2, 3
# CHECK: popcntd 2, 3                    # encoding: [0x7c,0x62,0x03,0xf4]
         popcntd 2, 3
# FIXME: bpermd 2, 3, 4

# Fixed-point rotate and shift instructions

# CHECK: rlwinm 2, 3, 4, 5, 6            # encoding: [0x54,0x62,0x21,0x4c]
         rlwinm 2, 3, 4, 5, 6
# CHECK: rlwinm. 2, 3, 4, 5, 6           # encoding: [0x54,0x62,0x21,0x4d]
         rlwinm. 2, 3, 4, 5, 6
# CHECK: rlwnm 2, 3, 4, 5, 6             # encoding: [0x5c,0x62,0x21,0x4c]
         rlwnm 2, 3, 4, 5, 6
# CHECK: rlwnm. 2, 3, 4, 5, 6            # encoding: [0x5c,0x62,0x21,0x4d]
         rlwnm. 2, 3, 4, 5, 6
# CHECK: rlwimi 2, 3, 4, 5, 6            # encoding: [0x50,0x62,0x21,0x4c]
         rlwimi 2, 3, 4, 5, 6
# CHECK: rlwimi. 2, 3, 4, 5, 6           # encoding: [0x50,0x62,0x21,0x4d]
         rlwimi. 2, 3, 4, 5, 6
# CHECK: rldicl 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x40]
         rldicl 2, 3, 4, 5
# CHECK: rldicl. 2, 3, 4, 5              # encoding: [0x78,0x62,0x21,0x41]
         rldicl. 2, 3, 4, 5
# CHECK: rldicr 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x44]
         rldicr 2, 3, 4, 5
# CHECK: rldicr. 2, 3, 4, 5              # encoding: [0x78,0x62,0x21,0x45]
         rldicr. 2, 3, 4, 5
# CHECK: rldic 2, 3, 4, 5                # encoding: [0x78,0x62,0x21,0x48]
         rldic 2, 3, 4, 5
# CHECK: rldic. 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x49]
         rldic. 2, 3, 4, 5
# CHECK: rldcl 2, 3, 4, 5                # encoding: [0x78,0x62,0x21,0x50]
         rldcl 2, 3, 4, 5
# CHECK: rldcl. 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x51]
         rldcl. 2, 3, 4, 5
# CHECK: rldcr 2, 3, 4, 5                # encoding: [0x78,0x62,0x21,0x52]
         rldcr 2, 3, 4, 5
# CHECK: rldcr. 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x53]
         rldcr. 2, 3, 4, 5
# CHECK: rldimi 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x4c]
         rldimi 2, 3, 4, 5
# CHECK: rldimi. 2, 3, 4, 5              # encoding: [0x78,0x62,0x21,0x4d]
         rldimi. 2, 3, 4, 5

# CHECK: slw 2, 3, 4                     # encoding: [0x7c,0x62,0x20,0x30]
         slw 2, 3, 4
# CHECK: slw. 2, 3, 4                    # encoding: [0x7c,0x62,0x20,0x31]
         slw. 2, 3, 4
# CHECK: srw 2, 3, 4                     # encoding: [0x7c,0x62,0x24,0x30]
         srw 2, 3, 4
# CHECK: srw. 2, 3, 4                    # encoding: [0x7c,0x62,0x24,0x31]
         srw. 2, 3, 4
# CHECK: srawi 2, 3, 4                   # encoding: [0x7c,0x62,0x26,0x70]
         srawi 2, 3, 4
# CHECK: srawi. 2, 3, 4                  # encoding: [0x7c,0x62,0x26,0x71]
         srawi. 2, 3, 4
# CHECK: sraw 2, 3, 4                    # encoding: [0x7c,0x62,0x26,0x30]
         sraw 2, 3, 4
# CHECK: sraw. 2, 3, 4                   # encoding: [0x7c,0x62,0x26,0x31]
         sraw. 2, 3, 4
# CHECK: sld 2, 3, 4                     # encoding: [0x7c,0x62,0x20,0x36]
         sld 2, 3, 4
# CHECK: sld. 2, 3, 4                    # encoding: [0x7c,0x62,0x20,0x37]
         sld. 2, 3, 4
# CHECK: srd 2, 3, 4                     # encoding: [0x7c,0x62,0x24,0x36]
         srd 2, 3, 4
# CHECK: srd. 2, 3, 4                    # encoding: [0x7c,0x62,0x24,0x37]
         srd. 2, 3, 4
# CHECK: sradi 2, 3, 4                   # encoding: [0x7c,0x62,0x26,0x74]
         sradi 2, 3, 4
# CHECK: sradi. 2, 3, 4                  # encoding: [0x7c,0x62,0x26,0x75]
         sradi. 2, 3, 4
# CHECK: srad 2, 3, 4                    # encoding: [0x7c,0x62,0x26,0x34]
         srad 2, 3, 4
# CHECK: srad. 2, 3, 4                   # encoding: [0x7c,0x62,0x26,0x35]
         srad. 2, 3, 4

# FIXME: BCD assist instructions

# Move to/from system register instructions

# FIXME: mtspr 256, 2
# FIXME: mfspr 2, 256
# CHECK: mtcrf 16, 2                     # encoding: [0x7c,0x41,0x01,0x20]
         mtcrf 16, 2
# CHECK: mfcr 2                          # encoding: [0x7c,0x40,0x00,0x26]
         mfcr 2
# FIXME: mtocrf 16, 2
# CHECK: mfocrf 16, 8                    # encoding: [0x7e,0x10,0x80,0x26]
         mfocrf 16, 8
# FIXME: mcrxr 2

