; Test to make sure that bss sections are printed with '.section' directive.
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic | FileCheck %s -check-prefix=PIC

@A = global i32 0

; CHECK:  .section  .bss,"aw",@nobits
; CHECK:  .globl A

; PIC:    .section  .got2,"aw",@progbits
; PIC:    .section  .bss,"aw",@nobits
; PIC:    .globl A
