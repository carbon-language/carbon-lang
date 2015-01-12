# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding \
# RUN: -mattr=micromips | FileCheck %s -check-prefix=CHECK-FIXUP
# RUN: llvm-mc %s -filetype=obj -triple=mipsel-unknown-linux \
# RUN: -mattr=micromips | llvm-readobj -r \
# RUN: | FileCheck %s -check-prefix=CHECK-ELF
#------------------------------------------------------------------------------
# Check that the assembler can handle the documented syntax
# for relocations.
#------------------------------------------------------------------------------
# CHECK-FIXUP: beqz16 $6, bar  # encoding: [0b0AAAAAAA,0x8f]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC7_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
# CHECK-FIXUP: bnez16 $6, bar  # encoding: [0b0AAAAAAA,0xaf]
# CHECK-FIXUP:                 #   fixup A - offset: 0,
# CHECK-FIXUP:                     value: bar, kind: fixup_MICROMIPS_PC7_S1
# CHECK-FIXUP: nop             # encoding: [0x00,0x00,0x00,0x00]
#------------------------------------------------------------------------------
# Check that the appropriate relocations were created.
#------------------------------------------------------------------------------
# CHECK-ELF: Relocations [
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC7_S1
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_PC7_S1
# CHECK-ELF: ]

  beqz16 $6, bar
  bnez16 $6, bar
