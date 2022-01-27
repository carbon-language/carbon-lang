# RUN: llvm-mc -triple=armv7-linux-gnueabi %s | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc -triple=armv7eb-linux-gnueabi %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple=armv7-linux-gnueabi %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s
# RUN: llvm-readelf -x .data %t | FileCheck --check-prefix=HEX %s

# RUN: llvm-mc -filetype=obj -triple=armv7eb-linux-gnueabi %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s
# RUN: llvm-readelf -x .data %t | FileCheck --check-prefix=HEX %s

.text
  bx lr
  nop
  nop
  .reloc 8, R_ARM_NONE, .data
  .reloc 4, R_ARM_NONE, foo+4
  .reloc 0, R_ARM_NONE, 8

  .reloc 0, R_ARM_ALU_PC_G0, .data+2
  .reloc 0, R_ARM_LDR_PC_G0, foo+3
  .reloc 0, R_ARM_THM_ALU_PREL_11_0, 5

  .reloc 0, BFD_RELOC_NONE, 9
  .reloc 0, BFD_RELOC_8, 9
  .reloc 0, BFD_RELOC_16, 9
  .reloc 0, BFD_RELOC_32, 9

.data
.globl foo
foo:
  .word 0
  .word 0
  .word 0

# PRINT: .reloc 8, R_ARM_NONE, .data
# PRINT: .reloc 4, R_ARM_NONE, foo+4
# PRINT: .reloc 0, R_ARM_NONE, 8
# PRINT: .reloc 0, R_ARM_ALU_PC_G0, .data+2
# PRINT: .reloc 0, R_ARM_LDR_PC_G0, foo+3
# PRINT: .reloc 0, R_ARM_THM_ALU_PREL_11_0, 5
# PRINT:      .reloc 0, BFD_RELOC_NONE, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_8, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_16, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_32, 9

# ARM relocations use the Elf32_Rel format. Addends are neither stored in the
# relocation entries nor applied in the referenced locations.
# CHECK:      0x8 R_ARM_NONE .data
# CHECK-NEXT: 0x4 R_ARM_NONE foo
# CHECK-NEXT: 0x0 R_ARM_NONE -
# CHECK-NEXT: 0x0 R_ARM_ALU_PC_G0 .data
# CHECK-NEXT: 0x0 R_ARM_LDR_PC_G0 foo
# CHECK-NEXT: 0x0 R_ARM_THM_ALU_PREL_11_0 -
# CHECK-NEXT: 0x0 R_ARM_NONE -
# CHECK-NEXT: 0x0 R_ARM_ABS8 -
# CHECK-NEXT: 0x0 R_ARM_ABS16 -
# CHECK-NEXT: 0x0 R_ARM_ABS32 -

# HEX: 0x00000000 00000000 00000000
