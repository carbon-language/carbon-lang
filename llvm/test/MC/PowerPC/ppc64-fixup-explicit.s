
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# RUN: llvm-mc -triple powerpc64-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r | FileCheck %s -check-prefix=REL

# GOT references must result in explicit relocations
# even if the target symbol is local.

target:

# CHECK: addi 4, 3, target@GOT           # encoding: [0x38,0x83,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@GOT, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16 target 0x0
         addi 4, 3, target@got  

# CHECK: ld 1, target@GOT(2)             # encoding: [0xe8,0x22,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@GOT, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_DS target 0x0
         ld 1, target@got(2)

# CHECK: addis 3, 2, target@got@ha       # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@ha, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_HA target 0x0
         addis 3, 2, target@got@ha

# CHECK: addi 4, 3, target@got@l         # encoding: [0x38,0x83,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_LO target 0x0
         addi 4, 3, target@got@l

# CHECK: addis 3, 2, target@got@h        # encoding: [0x3c,0x62,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@h, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_HI target 0x0
         addis 3, 2, target@got@h

# CHECK: lwz 1, target@got@l(3)          # encoding: [0x80,0x23,A,A]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@l, kind: fixup_ppc_half16
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_LO target 0x0
         lwz 1, target@got@l(3)

# CHECK: ld 1, target@got@l(3)           # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-NEXT:                            #   fixup A - offset: 2, value: target@got@l, kind: fixup_ppc_half16ds
# CHECK-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_LO_DS target 0x0
         ld 1, target@got@l(3)

