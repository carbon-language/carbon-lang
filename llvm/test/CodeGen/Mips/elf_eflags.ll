; This tests for directives that will result in
; ELF EFLAGS setting with direct object.

; Non-shared (static) is the absence of pic and or cpic.

; EF_MIPS_NOREORDER (0x00000001) is always on by default currently
; EF_MIPS_PIC (0x00000002)
; EF_MIPS_CPIC (0x00000004) - See note below
; EF_MIPS_ABI2 (0x00000020) - n32 not tested yet
; EF_MIPS_ARCH_32 (0x50000000)
; EF_MIPS_ARCH_64 (0x60000000)
; EF_MIPS_ARCH_32R2 (0x70000000)
; EF_MIPS_ARCH_64R2 (0x80000000)

; Note that EF_MIPS_CPIC is set by -mabicalls which is the default on Linux
; TODO need to support -mno-abicalls

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32 \
; RUN: -relocation-model=static %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-BE32 %s
;
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32 %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-BE32_PIC %s
;
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 \
; RUN: -relocation-model=static %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-BE32R2 %s
;
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-BE32R2_PIC %s
;
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 \
; RUN: -mattr=+micromips -relocation-model=static %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-BE32R2-MICROMIPS %s
;
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 \
; RUN: -mattr=+micromips %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-BE32R2-MICROMIPS_PIC %s

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=static %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-BE64 %s
;
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64 %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-BE64_PIC %s
;
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64r2 \
; RUN: -relocation-model=static %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-BE64R2 %s
;
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64r2 %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-BE64R2_PIC %s

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 \
; RUN: -mattr=+mips16 -relocation-model=pic %s -o - | \
; RUN: FileCheck -check-prefix=CHECK-LE32R2-MIPS16 %s

; 32(R1) bit with NO_REORDER and static
; CHECK-BE32: .abicalls
; CHECK-BE32: .option pic0
; CHECK-BE32: .set noreorder
; TODO: Need .set mips32
;
; 32(R1) bit with NO_REORDER and PIC
; CHECK-BE32_PIC: .abicalls
; CHECK-BE32_PIC: .set noreorder
; TODO: Need .set mips32 and check absence of .option pic0
;
; 32R2 bit with NO_REORDER and static
; CHECK-BE32R2: .abicalls
; CHECK-BE32R2: .option pic0
; CHECK-BE32R2: .set noreorder
; TODO: Need .set mips32r2
;
; 32R2 bit with NO_REORDER and PIC
; CHECK-BE32R2_PIC:.abicalls
; CHECK-BE32R2_PIC:.set noreorder
; TODO: Need .set mips32r2 and check absence of .option pic0
;
; 32R2 bit MICROMIPS with NO_REORDER and static
; CHECK-BE32R2-MICROMIPS: .abicalls
; CHECK-BE32R2-MICROMIPS: .option pic0
; CHECK-BE32R2-MICROMIPS: .set micromips
; CHECK-BE32R2-MICROMIPS: .set noreorder
; TODO: Need .set mips32r2
;
; 32R2 bit MICROMIPS with NO_REORDER and PIC
; CHECK-BE32R2-MICROMIPS_PIC: .abicalls
; CHECK-BE32R2-MICROMIPS_PIC: .set micromips
; CHECK-BE32R2-MICROMIPS_PIC: .set noreorder
; TODO: Need .set mips32r2 and check absence of .option pic0
;
; 64(R1) bit with NO_REORDER and static
; CHECK-BE64: .abicalls
; CHECK-BE64: .set noreorder
; TODO: Need .set mips64 and .option pic0
;
; 64(R1) bit with NO_REORDER and PIC
; CHECK-BE64_PIC: .abicalls
; CHECK-BE64_PIC: .set noreorder
; TODO: Need .set mips64 and check absence of .option pic0
;
; 64R2 bit with NO_REORDER and static
; CHECK-BE64R2: .abicalls
; CHECK-BE64R2: .set noreorder
; TODO: Need .set mips64r2 and .option pic0
;
; 64R2 bit with NO_REORDER and PIC
; CHECK-BE64R2_PIC: .abicalls
; CHECK-BE64R2_PIC: .set noreorder
; TODO: Need .set mips64r2 and check absence of .option pic0
;
; 32R2 bit MIPS16 with PIC
; CHECK-LE32R2-MIPS16: .abicalls
; CHECK-LE32R2-MIPS16: .set mips16
; TODO: Need .set mips32r2 and check absence of .option pic0 and noreorder

define i32 @main() nounwind {
entry:
  ret i32 0
}
