; 
; When the assembler is ready a .s file for it will
; be created.

; Note that EF_MIPS_CPIC is set by -mabicalls which is the default on Linux
; TODO need to support -mno-abicalls

; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 -relocation-model=static %s -o - | FileCheck -check-prefix=CHECK-STATIC %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 %s -o - | FileCheck -check-prefix=CHECK-PIC %s
; RUN: llc -filetype=asm -mtriple mips64el-unknown-linux -mcpu=mips4 -relocation-model=static %s -o - | FileCheck -check-prefix=CHECK-PIC %s
; RUN: llc -filetype=asm -mtriple mips64el-unknown-linux -mcpu=mips64 -relocation-model=static %s -o - | FileCheck -check-prefix=CHECK-PIC %s

; CHECK-STATIC: .abicalls
; CHECK-STATIC-NEXT: pic0
; CHECK-PIC: .abicalls
; CHECK-PIC-NOT: pic0
