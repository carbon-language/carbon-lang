# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for control instructions.
#------------------------------------------------------------------------------
# microMIPS Control Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL:    break                      # encoding: [0x00,0x00,0x07,0x00]
# CHECK-EL:    break 7, 0                 # encoding: [0x07,0x00,0x07,0x00]
# CHECK-EL:    break 7, 5                 # encoding: [0x07,0x00,0x47,0x01]
# CHECK-EL:    syscall                    # encoding: [0x00,0x00,0xfc,0x8a]
# CHECK-EL:    syscall 13396              # encoding: [0x54,0x00,0xfc,0x8a]
# CHECK-EL:    eret                       # encoding: [0x00,0x00,0x7c,0xf3]
# CHECK-EL:    deret                      # encoding: [0x00,0x00,0x7c,0xe3]
# CHECK-EL:    di                         # encoding: [0x00,0x00,0x7c,0x47]
# CHECK-EL:    di                         # encoding: [0x00,0x00,0x7c,0x47]
# CHECK-EL:    di  $10                    # encoding: [0x0a,0x00,0x7c,0x47]
# CHECK-EL:    ei                         # encoding: [0x00,0x00,0x7c,0x57]
# CHECK-EL:    ei                         # encoding: [0x00,0x00,0x7c,0x57]
# CHECK-EL:    ei  $10                    # encoding: [0x0a,0x00,0x7c,0x57]
# CHECK-EL:    wait                       # encoding: [0x00,0x00,0x7c,0x93]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB:   break                       # encoding: [0x00,0x00,0x00,0x07]
# CHECK-EB:   break 7, 0                  # encoding: [0x00,0x07,0x00,0x07]
# CHECK-EB:   break 7, 5                  # encoding: [0x00,0x07,0x01,0x47]
# CHECK-EB:   syscall                     # encoding: [0x00,0x00,0x8a,0xfc]
# CHECK-EB:   syscall 13396               # encoding: [0x00,0x54,0x8a,0xfc]
# CHECK-EB:   eret                        # encoding: [0x00,0x00,0xf3,0x7c]
# CHECK-EB:   deret                       # encoding: [0x00,0x00,0xe3,0x7c]
# CHECK-EB:   di                          # encoding: [0x00,0x00,0x47,0x7c]
# CHECK-EB:   di                          # encoding: [0x00,0x00,0x47,0x7c]
# CHECK-EB:   di  $10                     # encoding: [0x00,0x0a,0x47,0x7c]
# CHECK-EB:   ei                          # encoding: [0x00,0x00,0x57,0x7c]
# CHECK-EB:   ei                          # encoding: [0x00,0x00,0x57,0x7c]
# CHECK-EB:   ei  $10                     # encoding: [0x00,0x0a,0x57,0x7c]
# CHECK-EB:   wait                        # encoding: [0x00,0x00,0x93,0x7c]

    break
    break 7
    break 7,5
    syscall
    syscall 0x3454
    eret
    deret
    di
    di $0
    di $10
    ei
    ei $0
    ei $10
    wait
