# REQUIRES: mips
# Check MIPS ELF ISA flag calculation if input files have different ISAs.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32 %S/Inputs/mips-dynamic.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r2 %s -o %t2.o
# RUN: ld.lld %t1.o %t2.o -o %t.exe
# RUN: llvm-readobj -h %t.exe | FileCheck -check-prefix=R1R2 %s

# Check that lld does not allow to link incompatible ISAs.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips3 %S/Inputs/mips-dynamic.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32 -mattr=+fp64 %s -o %t2.o
# RUN: not ld.lld %t1.o %t2.o -o /dev/null 2>&1 | FileCheck -check-prefix=R3R32 %s

# Check that lld does not allow to link incompatible ISAs.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         -mcpu=mips64r6 %S/Inputs/mips-dynamic.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         -position-independent -mcpu=octeon %s -o %t2.o
# RUN: not ld.lld %t1.o %t2.o -o /dev/null 2>&1 \
# RUN:   | FileCheck -check-prefix=R6OCTEON %s

# Check that lld take in account EF_MIPS_MACH_XXX ISA flags

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         -position-independent -mcpu=mips64 %S/Inputs/mips-dynamic.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         -position-independent -mcpu=octeon %s -o %t2.o
# RUN: ld.lld %t1.o %t2.o -o %t.exe
# RUN: llvm-readobj -h %t.exe | FileCheck -check-prefix=OCTEON %s

  .option pic0
  .text
  .global  __start
__start:
  nop

# R1R2:      Flags [
# R1R2-NEXT:   EF_MIPS_ABI_O32
# R1R2-NEXT:   EF_MIPS_ARCH_32R2
# R1R2-NEXT:   EF_MIPS_CPIC
# R1R2-NEXT: ]

# R3R32:      error: incompatible target ISA:
# R3R32-NEXT: >>> {{.+}}mips-elf-flags-err.s.tmp1.o: mips3
# R3R32-NEXT: >>> {{.+}}mips-elf-flags-err.s.tmp2.o: mips32

# R6OCTEON:      error: incompatible target ISA:
# R6OCTEON-NEXT: >>> {{.+}}mips-elf-flags-err.s.tmp1.o: mips64r6
# R6OCTEON-NEXT: >>> {{.+}}mips-elf-flags-err.s.tmp2.o: mips64r2 (octeon)

# OCTEON:      Flags [
# OCTEON-NEXT:   EF_MIPS_ARCH_64R2
# OCTEON-NEXT:   EF_MIPS_CPIC
# OCTEON-NEXT:   EF_MIPS_MACH_OCTEON
# OCTEON:      ]
