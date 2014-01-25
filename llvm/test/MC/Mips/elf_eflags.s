// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o - | \
// RUN: llvm-readobj -h | FileCheck %s

// From the commandline and defaults the following should be set:
//   EF_MIPS_ARCH_32      (0x50000000)
//   EF_MIPS_ABI_O32      (0x00001000)
//   EF_MIPS_NOREORDER    (0x00000001)
//   EF_MIPS_PIC          (0x00000002)

// Inline directives should set or unset the following:
//   EF_MIPS_CPIC         (0x00000004) : .abicalls
//   EF_MIPS_ARCH_ASE_M16 (0x04000000) : .set mips16
//   The negation of EF_MIPS_PIC : .option pic0

// CHECK: Flags [ (0x54001005)

        .abicalls

        .option pic0

        .set mips16
