// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o -| llvm-readobj -h | FileCheck %s
// The initial value will be set at 0x50001003 and
// we will override that with the negation of 0x2 (option pic0
// the addition of 0x4 (.abicalls)

        .mips_hack_elf_flags 0x50001003

// CHECK: Flags [ (0x54001005)

        .abicalls

        .option pic0
 
 // Set EF_MIPS_ARCH_ASE_M16 (0x04000000)
        .set mips16
