# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding \
# RUN: -mattr=micromips | FileCheck %s -check-prefix=CHECK-FIXUP
# RUN: llvm-mc %s -filetype=obj -triple=mipsel-unknown-linux \
# RUN: -mattr=micromips | llvm-readobj -r \
# RUN: | FileCheck %s -check-prefix=CHECK-ELF
#------------------------------------------------------------------------------
# Check that the assembler can handle the documented syntax
# for relocations.
#------------------------------------------------------------------------------
# CHECK-FIXUP: b           bar # encoding: [A,0x94'A',0x00,0x00]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC16_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
# CHECK-FIXUP: beq $3, $4, bar # encoding: [0x83'A',0x94'A',0x00,0x00]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC16_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
# CHECK-FIXUP: bne $3, $4, bar # encoding: [0x83'A',0xb4'A',0x00,0x00]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC16_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
# CHECK-FIXUP: bgez    $4, bar # encoding: [0x44'A',0x40'A',0x00,0x00]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC16_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
# CHECK-FIXUP: bgtz    $4, bar # encoding: [0xc4'A',0x40'A',0x00,0x00]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC16_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
# CHECK-FIXUP: blez    $4, bar # encoding: [0x84'A',0x40'A',0x00,0x00]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC16_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
# CHECK-FIXUP: bltz    $4, bar # encoding: [0x04'A',0x40'A',0x00,0x00]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC16_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
# CHECK-FIXUP: bgezal  $4, bar # encoding: [0x64'A',0x40'A',0x00,0x00]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC16_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
# CHECK-FIXUP: bltzal  $4, bar # encoding: [0x24'A',0x40'A',0x00,0x00]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC16_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
#------------------------------------------------------------------------------
# Check that the appropriate relocations were created.
#------------------------------------------------------------------------------
# CHECK-ELF: Relocations [
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC16_S1
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC16_S1
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC16_S1
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC16_S1
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC16_S1
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC16_S1
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC16_S1
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC16_S1
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC16_S1
# CHECK-ELF: ]

  b       bar
  beq     $3, $4, bar
  bne     $3, $4, bar
  bgez    $4, bar
  bgtz    $4, bar
  blez    $4, bar
  bltz    $4, bar
  bgezal  $4, bar
  bltzal  $4, bar
