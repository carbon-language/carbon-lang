# RUN: llvm-mc -triple mips64-unknown-unknown -mattr=-n64,+o32 -filetype=obj -o - %s | \
# RUN:   llvm-objdump -d -r -arch=mips64 - | \
# RUN:     FileCheck -check-prefix=O32 %s

# RUN: llvm-mc -triple mips64-unknown-unknown -mattr=-n64,+o32 %s | \
# RUN:   FileCheck -check-prefix=ASM %s

# RUN: llvm-mc -triple mips64-unknown-unknown -mattr=-n64,+n32 -filetype=obj -o - %s | \
# RUN:   llvm-objdump -d -r -arch=mips64 - | \
# RUN:     FileCheck -check-prefix=NXX -check-prefix=N32 %s

# RUN: llvm-mc -triple mips64-unknown-unknown -mattr=-n64,+n32 %s | \
# RUN:   FileCheck -check-prefix=ASM %s

# RUN: llvm-mc -triple mips64-unknown-unknown %s -filetype=obj -o - | \
# RUN:   llvm-objdump -d -r -arch=mips64 - | \
# RUN:     FileCheck -check-prefix=NXX -check-prefix=N64 %s

# RUN: llvm-mc -triple mips64-unknown-unknown %s | \
# RUN:   FileCheck -check-prefix=ASM %s

        .text
        .option pic2
t1:
        .cpsetup $25, 8, __cerror


# O32-NOT: __cerror

# FIXME: Direct object emission for N32 is still under development.
# N32 doesn't allow 3 operations to be specified in the same relocation
# record like N64 does.

# NXX: sd       $gp, 8($sp)
# NXX: lui      $gp, 0
# NXX: R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_HI16  __cerror
# NXX: addiu    $gp, $gp, 0
# NXX: R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_LO16  __cerror
# N32: addu     $gp, $gp, $25
# N64: daddu    $gp, $gp, $25

# ASM: .cpsetup $25, 8, __cerror

t2:

        .cpsetup $25, $2, __cerror

# O32-NOT: __cerror

# FIXME: Direct object emission for N32 is still under development.
# N32 doesn't allow 3 operations to be specified in the same relocation
# record like N64 does.

# NXX: move     $2, $gp
# NXX: lui      $gp, 0
# NXX: R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_HI16  __cerror
# NXX: addiu    $gp, $gp, 0
# NXX: R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_LO16  __cerror
# N32: addu     $gp, $gp, $25
# N64: daddu    $gp, $gp, $25

# ASM: .cpsetup $25, $2, __cerror

t3:
        .option pic0
        nop
        .cpsetup $25, 8, __cerror
        nop

# Testing that .cpsetup expands to nothing in this case
# by checking that the next instruction after the first
# nop is also a 'nop'.
# NXX: nop
# NXX-NEXT: nop

# ASM: nop
# ASM: .cpsetup $25, 8, __cerror
# ASM: nop
