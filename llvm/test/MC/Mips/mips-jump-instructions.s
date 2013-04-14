# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN: FileCheck %s
# Check that the assembler can handle the documented syntax
# for jumps and branches.
#------------------------------------------------------------------------------
# Branch instructions
#------------------------------------------------------------------------------
# CHECK:   b 1332                 # encoding: [0x4d,0x01,0x00,0x10]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bc1f 1332              # encoding: [0x4d,0x01,0x00,0x45]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bc1t 1332              # encoding: [0x4d,0x01,0x01,0x45]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   beq $9, $6, 1332       # encoding: [0x4d,0x01,0x26,0x11]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bgez $6, 1332          # encoding: [0x4d,0x01,0xc1,0x04]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bgezal $6, 1332        # encoding: [0x4d,0x01,0xd1,0x04]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bgtz $6, 1332          # encoding: [0x4d,0x01,0xc0,0x1c]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   blez $6, 1332          # encoding: [0x4d,0x01,0xc0,0x18]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bne $9, $6, 1332       # encoding: [0x4d,0x01,0x26,0x15]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bal     1332           # encoding: [0x4d,0x01,0x11,0x04]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]

.set noreorder

         b 1332
         nop
         bc1f 1332
         nop
         bc1t 1332
         nop
         beq $9,$6,1332
         nop
         bgez $6,1332
         nop
         bgezal $6,1332
         nop
         bgtz $6,1332
         nop
         blez $6,1332
         nop
         bne $9,$6,1332
         nop
         bal 1332
         nop

end_of_code:
#------------------------------------------------------------------------------
# Jump instructions
#------------------------------------------------------------------------------
# CHECK:   j 1328               # encoding: [0x4c,0x01,0x00,0x08]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jal 1328             # encoding: [0x4c,0x01,0x00,0x0c]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jalr $6              # encoding: [0x09,0xf8,0xc0,0x00]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jalr $25             # encoding: [0x09,0xf8,0x20,0x03]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jalr $10, $11        # encoding: [0x09,0x50,0x60,0x01]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jr $7                # encoding: [0x08,0x00,0xe0,0x00]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jr $7                # encoding: [0x08,0x00,0xe0,0x00]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jalr  $25            # encoding: [0x09,0xf8,0x20,0x03]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jalr  $4, $25        # encoding: [0x09,0x20,0x20,0x03]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]


   j 1328
   nop
   jal 1328
   nop
   jalr $6
   nop
   jalr $31, $25
   nop
   jalr $10, $11
   nop
   jr $7
   nop
   j $7
   nop
   jal  $25
   nop
   jal  $4,$25
   nop
