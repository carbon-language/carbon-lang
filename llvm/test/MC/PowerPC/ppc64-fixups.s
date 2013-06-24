
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# RUN: llvm-mc -triple powerpc64-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r | FileCheck %s -check-prefix=REL

# CHECK: b target                        # encoding: [0b010010AA,A,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24
# CHECK-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_REL24 target 0x0
         b target

# CHECK: ba target                       # encoding: [0b010010AA,A,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24abs
# CHECK-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR24 target 0x0
         ba target

# CHECK: beq 0, target                   # encoding: [0x41,0x82,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_REL14 target 0x0
         beq target

# CHECK: beqa 0, target                  # encoding: [0x41,0x82,A,0bAAAAAA10]
# CHECK-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR14 target 0x0
         beqa target


# FIXME: .TOC.@tocbase

# CHECK: li 3, target@l                  # encoding: [0x38,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO target 0x0
         li 3, target@l

# CHECK: addis 3, 3, target@ha           # encoding: [0x3c,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HA target 0x0
         addis 3, 3, target@ha

# CHECK: lis 3, target@ha                # encoding: [0x3c,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HA target 0x0
         lis 3, target@ha

# CHECK: addi 4, 3, target@l             # encoding: [0x38,0x83,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO target 0x0
         addi 4, 3, target@l

# CHECK: li 3, target@ha                 # encoding: [0x38,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HA target 0x0
         li 3, target@ha

# CHECK: lis 3, target@l                 # encoding: [0x3c,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO target 0x0
         lis 3, target@l

# CHECK: li 3, target                    # encoding: [0x38,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16 target 0x0
         li 3, target

# CHECK: lis 3, target                   # encoding: [0x3c,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16 target 0x0
         lis 3, target

# CHECK: li 3, target@h                  # encoding: [0x38,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HI target 0x0
         li 3, target@h

# CHECK: lis 3, target@h                  # encoding: [0x3c,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HI target 0x0
         lis 3, target@h

# CHECK: li 3, target@higher             # encoding: [0x38,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@higher, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HIGHER target 0x0
         li 3, target@higher

# CHECK: lis 3, target@highest           # encoding: [0x3c,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@highest, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HIGHEST target 0x0
         lis 3, target@highest

# CHECK: li 3, target@highera            # encoding: [0x38,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@highera, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HIGHERA target 0x0
         li 3, target@highera

# CHECK: lis 3, target@highesta          # encoding: [0x3c,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@highesta, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HIGHESTA target 0x0
         lis 3, target@highesta

# CHECK: lwz 1, target@l(3)              # encoding: [0x80,0x23,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO target 0x0
         lwz 1, target@l(3)

# CHECK: ld 1, target@l(3)               # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO_DS target 0x0
         ld 1, target@l(3)

# CHECK: ld 1, target(3)                 # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_DS target 0x0
         ld 1, target(3)

base:
# CHECK: li 3, target-base               # encoding: [0x38,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target-base, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_REL16 target 0x2
         li 3, target-base

# CHECK: li 3, target-base@h             # encoding: [0x38,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target-base@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_REL16_HI target 0x6
         li 3, target-base@h

# CHECK: li 3, target-base@l             # encoding: [0x38,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target-base@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_REL16_LO target 0xA
         li 3, target-base@l

# CHECK: li 3, target-base@ha            # encoding: [0x38,0x60,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target-base@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_REL16_HA target 0xE
         li 3, target-base@ha

# CHECK: ld 1, target@toc(2)             # encoding: [0xe8,0x22,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@toc, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_DS target 0x0
         ld 1, target@toc(2)

# CHECK: addis 3, 2, target@toc@ha       # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@toc@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_HA target 0x0
         addis 3, 2, target@toc@ha

# CHECK: addi 4, 3, target@toc@l         # encoding: [0x38,0x83,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@toc@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_LO target 0x0
         addi 4, 3, target@toc@l

# CHECK: addis 3, 2, target@toc@h        # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@toc@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_HI target 0x0
         addis 3, 2, target@toc@h

# CHECK: lwz 1, target@toc@l(3)          # encoding: [0x80,0x23,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@toc@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_LO target 0x0
         lwz 1, target@toc@l(3)

# CHECK: ld 1, target@toc@l(3)           # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@toc@l, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_LO_DS target 0x0
         ld 1, target@toc@l(3)

# FIXME: @tls


# CHECK: addis 3, 2, target@tprel@ha     # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@tprel@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HA target 0x0
         addis 3, 2, target@tprel@ha

# CHECK: addi 3, 3, target@tprel@l       # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@tprel@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_LO target 0x0
         addi 3, 3, target@tprel@l

# CHECK: addi 3, 3, target@tprel         # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@tprel, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16 target 0x0
         addi 3, 3, target@tprel

# CHECK: addi 3, 3, target@tprel@h       # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@tprel@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HI target 0x0
         addi 3, 3, target@tprel@h

# CHECK: addi 3, 3, target@tprel@higher  # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@tprel@higher, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HIGHER target 0x0
         addi 3, 3, target@tprel@higher

# CHECK: addis 3, 2, target@tprel@highest # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@tprel@highest, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HIGHEST target 0x0
         addis 3, 2, target@tprel@highest

# CHECK: addi 3, 3, target@tprel@highera  # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@tprel@highera, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HIGHERA target 0x0
         addi 3, 3, target@tprel@highera

# CHECK: addis 3, 2, target@tprel@highesta # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@tprel@highesta, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HIGHESTA target 0x0
         addis 3, 2, target@tprel@highesta

# CHECK: ld 1, target@tprel@l(3)         # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@tprel@l, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_LO_DS target 0x0
         ld 1, target@tprel@l(3)

# CHECK: ld 1, target@tprel(3)           # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@tprel, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_DS target 0x0
         ld 1, target@tprel(3)

# CHECK: addis 3, 2, target@dtprel@ha    # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HA target 0x0
         addis 3, 2, target@dtprel@ha

# CHECK: addi 3, 3, target@dtprel@l      # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_LO target 0x0
         addi 3, 3, target@dtprel@l

# CHECK: addi 3, 3, target@dtprel         # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@dtprel, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16 target 0x0
         addi 3, 3, target@dtprel

# CHECK: addi 3, 3, target@dtprel@h       # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HI target 0x0
         addi 3, 3, target@dtprel@h

# CHECK: addi 3, 3, target@dtprel@higher  # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@higher, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HIGHER target 0x0
         addi 3, 3, target@dtprel@higher

# CHECK: addis 3, 2, target@dtprel@highest # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@highest, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HIGHEST target 0x0
         addis 3, 2, target@dtprel@highest

# CHECK: addi 3, 3, target@dtprel@highera  # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@highera, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HIGHERA target 0x0
         addi 3, 3, target@dtprel@highera

# CHECK: addis 3, 2, target@dtprel@highesta # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@highesta, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HIGHESTA target 0x0
         addis 3, 2, target@dtprel@highesta

# CHECK: ld 1, target@dtprel@l(3)        # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@l, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_LO_DS target 0x0
         ld 1, target@dtprel@l(3)

# CHECK: ld 1, target@dtprel(3)          # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@dtprel, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_DS target 0x0
         ld 1, target@dtprel(3)


# CHECK: addis 3, 2, target@got@tprel@ha # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tprel@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TPREL16_HA target 0x0
         addis 3, 2, target@got@tprel@ha

# CHECK: ld 1, target@got@tprel@l(3)     # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tprel@l, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TPREL16_LO_DS target 0x0
         ld 1, target@got@tprel@l(3)

# CHECK: addis 3, 2, target@got@tprel@h  # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tprel@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TPREL16_HI target 0x0
         addis 3, 2, target@got@tprel@h

# CHECK: ld 1, target@got@tprel(3)       # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tprel, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TPREL16_DS target 0x0
         ld 1, target@got@tprel(3)

# CHECK: addis 3, 2, target@got@dtprel@ha # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@dtprel@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_DTPREL16_HA target 0x0
         addis 3, 2, target@got@dtprel@ha

# CHECK: ld 1, target@got@dtprel@l(3)    # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@dtprel@l, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_DTPREL16_LO_DS target 0x0
         ld 1, target@got@dtprel@l(3)

# CHECK: addis 3, 2, target@got@dtprel@h # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@dtprel@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_DTPREL16_HI target 0x0
         addis 3, 2, target@got@dtprel@h

# CHECK: ld 1, target@got@dtprel(3)      # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@dtprel, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_DTPREL16_DS target 0x0
         ld 1, target@got@dtprel(3)

# CHECK: addis 3, 2, target@got@tlsgd@ha # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsgd@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSGD16_HA target 0x0
         addis 3, 2, target@got@tlsgd@ha

# CHECK: addi 3, 3, target@got@tlsgd@l   # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsgd@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSGD16_LO target 0x0
         addi 3, 3, target@got@tlsgd@l

# CHECK: addi 3, 3, target@got@tlsgd@h   # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsgd@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSGD16_HI target 0x0
         addi 3, 3, target@got@tlsgd@h

# CHECK: addi 3, 3, target@got@tlsgd     # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsgd, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSGD16 target 0x0
         addi 3, 3, target@got@tlsgd


# CHECK: addis 3, 2, target@got@tlsld@ha # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsld@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSLD16_HA target 0x0
         addis 3, 2, target@got@tlsld@ha

# CHECK: addi 3, 3, target@got@tlsld@l   # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsld@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSLD16_LO target 0x0
         addi 3, 3, target@got@tlsld@l

# CHECK: addi 3, 3, target@got@tlsld@h   # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsld@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSLD16_HI target 0x0
         addi 3, 3, target@got@tlsld@h

# CHECK: addi 3, 3, target@got@tlsld     # encoding: [0x38,0x63,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsld, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSLD16 target 0x0
         addi 3, 3, target@got@tlsld

