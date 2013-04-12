; This tests ELF EFLAGS setting with direct object.
; When the assembler is ready a .s file for it will
; be created.

; Non-shared (static) is the absence of pic and or cpic.

; EF_MIPS_NOREORDER (0x00000001) is always on by default currently
; EF_MIPS_PIC (0x00000002)
; EF_MIPS_CPIC (0x00000004) - not tested yet
; EF_MIPS_ABI2 (0x00000020) - n32 not tested yet
; EF_MIPS_ARCH_32 (0x50000000)
; EF_MIPS_ARCH_64 (0x60000000)
; EF_MIPS_ARCH_32R2 (0x70000000)
; EF_MIPS_ARCH_64R2 (0x80000000)

; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32 -relocation-model=static %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-BE32 %s
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32 %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-BE32_PIC %s
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32r2 -relocation-model=static %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-BE32R2 %s
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32r2 %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-BE32R2_PIC %s
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+micromips -relocation-model=static %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-BE32R2-MICROMIPS %s
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+micromips %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-BE32R2-MICROMIPS_PIC %s

; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips64 -relocation-model=static %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-BE64 %s
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips64 %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-BE64_PIC %s
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips64r2 -relocation-model=static %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-BE64R2 %s
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips64r2 %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-BE64R2_PIC %s

; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+mips16 -relocation-model=pic %s -o - | llvm-readobj -h | FileCheck -check-prefix=CHECK-LE32R2-MIPS16 %s
 
; 32(R1) bit with NO_REORDER and static
; CHECK-BE32: Flags [ (0x50001001)
;
; 32(R1) bit with NO_REORDER and PIC
; CHECK-BE32_PIC: Flags [ (0x50001003)
;
; 32R2 bit with NO_REORDER and static
; CHECK-BE32R2: Flags [ (0x70001001)
;
; 32R2 bit with NO_REORDER and PIC
; CHECK-BE32R2_PIC: Flags [ (0x70001003)
;
; 32R2 bit MICROMIPS with NO_REORDER and static
; CHECK-BE32R2-MICROMIPS: Flags [ (0x72001001)
;
; 32R2 bit MICROMIPS with NO_REORDER and PIC
;CHECK-BE32R2-MICROMIPS_PIC: Flags [ (0x72001003)
;
; 64(R1) bit with NO_REORDER and static
; CHECK-BE64: Flags [ (0x60000001)
;
; 64(R1) bit with NO_REORDER and PIC
; CHECK-BE64_PIC: Flags [ (0x60000003)
;
; 64R2 bit with NO_REORDER and static
; CHECK-BE64R2: Flags [ (0x80000001)
;
; 64R2 bit with NO_REORDER and PIC
; CHECK-BE64R2_PIC: Flags [ (0x80000003)
;
; 32R2 bit MIPS16 with PIC
; CHECK-LE32R2-MIPS16: Flags [ (0x74001002)
 
define i32 @main() nounwind {
entry:
  ret i32 0
}
