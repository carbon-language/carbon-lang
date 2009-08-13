; Test to make sure that bss sections are printed with '.section' directive.
; RUN: llvm-as < %s | llc -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

@A = global i32 0

; CHECK:  .section  .bss,"aw",@nobits
; CHECK:  .global A

