# RUN: llvm-mc -filetype=obj -arch mipsel %s | llvm-readobj -r | FileCheck %s

# Test the order of records in the relocation table.
# *HI16 and local *GOT16 relocations should be immediately followed by the
# corresponding *LO16 relocation against the same symbol.
#
# We try to implement the same semantics as gas, ie. to order the relocation
# table the same way as gas.
#
# gnu as command line:
# mips-linux-gnu-as -EL sort-relocation-table.s -o sort-relocation-table.o
#
# TODO: Add mips16 and micromips tests.
# Note: offsets are part of expected output, so it's simpler to add new test
#       cases at the bottom of the file.

# CHECK:       Relocations [
# CHECK-NEXT:  {

# Put HI before LO.
addiu $2,$2,%lo(sym1)
lui $2,%hi(sym1)

# CHECK-NEXT:    0x4 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x0 R_MIPS_LO16 sym1

# When searching for a matching LO, ignore LOs against a different symbol.
addiu $2,$2,%lo(sym2)
lui $2,%hi(sym2)
addiu $2,$2,%lo(sym2_d)

# CHECK-NEXT:    0xC R_MIPS_HI16 sym2
# CHECK-NEXT:    0x8 R_MIPS_LO16 sym2
# CHECK-NEXT:    0x10 R_MIPS_LO16 sym2_d

# Match HI with 2nd LO because it has higher offset (than the 1st LO).
addiu $2,$2,%lo(sym3)
addiu $2,$2,%lo(sym3)
lui $2,%hi(sym3)

# CHECK-NEXT:    0x14 R_MIPS_LO16 sym3
# CHECK-NEXT:    0x1C R_MIPS_HI16 sym3
# CHECK-NEXT:    0x18 R_MIPS_LO16 sym3

# HI is already followed by a matching LO, so don't look further, ie. ignore the
# "free" LO with higher offset.
lui $2,%hi(sym4)
addiu $2,$2,%lo(sym4)
addiu $2,$2,%lo(sym4)

# CHECK-NEXT:    0x20 R_MIPS_HI16 sym4
# CHECK-NEXT:    0x24 R_MIPS_LO16 sym4
# CHECK-NEXT:    0x28 R_MIPS_LO16 sym4

# Match 2nd HI with 2nd LO, since it's the one with highest offset among the
# "free" ones.
addiu $2,$2,%lo(sym5)
addiu $2,$2,%lo(sym5)
lui $2,%hi(sym5)
addiu $2,$2,%lo(sym5)
lui $2,%hi(sym5)

# CHECK-NEXT:    0x2C R_MIPS_LO16 sym5
# CHECK-NEXT:    0x3C R_MIPS_HI16 sym5
# CHECK-NEXT:    0x30 R_MIPS_LO16 sym5
# CHECK-NEXT:    0x34 R_MIPS_HI16 sym5
# CHECK-NEXT:    0x38 R_MIPS_LO16 sym5

# When more HIs are matched with one LO, sort them in descending order of
# offset.
addiu $2,$2,%lo(sym6)
lui $2,%hi(sym6)
lui $2,%hi(sym6)

# CHECK-NEXT:    0x48 R_MIPS_HI16 sym6
# CHECK-NEXT:    0x44 R_MIPS_HI16 sym6
# CHECK-NEXT:    0x40 R_MIPS_LO16 sym6

#  sym7 is a local symbol, so GOT relocation against it needs a matching LO.
sym7:
addiu $2,$2,%lo(sym7)
lui $2,%got(sym7)

# CHECK-NEXT:    0x50 R_MIPS_GOT16 sym7
# CHECK-NEXT:    0x4C R_MIPS_LO16 sym7

# sym8 is not a local symbol, don't look for a matching LO for GOT.
.global sym8
addiu $2,$2,%lo(sym8)
lui $2,%got(sym8)

# CHECK-NEXT:    0x54 R_MIPS_LO16 sym8
# CHECK-NEXT:    0x58 R_MIPS_GOT16 sym8

# A small combination of previous checks.
symc1:
addiu $2,$2,%lo(symc1)
addiu $2,$2,%lo(symc1)
addiu $2,$2,%lo(symc1)
lui $2,%hi(symc1)
lui $2,%got(symc1)
addiu $2,$2,%lo(symc2)
lui $2,%hi(symc1)
lui $2,%hi(symc1)
lui $2,%got(symc2)
lui $2,%hi(symc1)
addiu $2,$2,%lo(symc1)
addiu $2,$2,%lo(symc2)
lui $2,%hi(symc1)
lui $2,%hi(symc1)

# CHECK-NEXT:    0x78 R_MIPS_HI16 symc1
# CHECK-NEXT:    0x74 R_MIPS_HI16 symc1
# CHECK-NEXT:    0x6C R_MIPS_GOT16 symc1
# CHECK-NEXT:    0x68 R_MIPS_HI16 symc1
# CHECK-NEXT:    0x5C R_MIPS_LO16 symc1
# CHECK-NEXT:    0x8C R_MIPS_HI16 symc1
# CHECK-NEXT:    0x60 R_MIPS_LO16 symc1
# CHECK-NEXT:    0x90 R_MIPS_HI16 symc1
# CHECK-NEXT:    0x64 R_MIPS_LO16 symc1
# CHECK-NEXT:    0x70 R_MIPS_LO16 symc2
# CHECK-NEXT:    0x7C R_MIPS_GOT16 symc2
# CHECK-NEXT:    0x80 R_MIPS_HI16 symc1
# CHECK-NEXT:    0x84 R_MIPS_LO16 symc1
# CHECK-NEXT:    0x88 R_MIPS_LO16 symc2
