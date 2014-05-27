# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64r6 \
# RUN:   | FileCheck %s -check-prefix=CHECK-FIXUP
# RUN: llvm-mc %s -filetype=obj -triple=mips-unknown-linux -mcpu=mips64r6 \
# RUN:   | llvm-readobj -r | FileCheck %s -check-prefix=CHECK-ELF
#------------------------------------------------------------------------------
# Check that the assembler can handle the documented syntax for fixups.
#------------------------------------------------------------------------------
# CHECK-FIXUP: beqc $5, $6, bar # encoding: [0x20,0xa6,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_Mips_PC16
# CHECK-FIXUP: bnec $5, $6, bar # encoding: [0x60,0xa6,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_Mips_PC16
# CHECK-FIXUP: beqzc $9, bar    # encoding: [0xd9,0b001AAAAA,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_MIPS_PC21_S2
# CHECK-FIXUP: bnezc $9, bar    # encoding: [0xf9,0b001AAAAA,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_MIPS_PC21_S2
# CHECK-FIXUP: balc  bar        # encoding: [0b111010AA,A,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_MIPS_PC26_S2
# CHECK-FIXUP: bc    bar        # encoding: [0b110010AA,A,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_MIPS_PC26_S2
#------------------------------------------------------------------------------
# Check that the appropriate relocations were created.
#------------------------------------------------------------------------------
# CHECK-ELF: Relocations [
# CHECK-ELF:     0x0 R_MIPS_PC16 bar 0x0
# CHECK-ELF:     0x4 R_MIPS_PC16 bar 0x0
# CHECK-ELF:     0x8 R_MIPS_PC21_S2 bar 0x0
# CHECK-ELF:     0xC R_MIPS_PC21_S2 bar 0x0
# CHECK-ELF:     0x10 R_MIPS_PC26_S2 bar 0x0
# CHECK-ELF:     0x14 R_MIPS_PC26_S2 bar 0x0
# CHECK-ELF: ]

  beqc  $5, $6, bar
  bnec  $5, $6, bar
  beqzc $9, bar
  bnezc $9, bar
  balc  bar
  bc    bar
