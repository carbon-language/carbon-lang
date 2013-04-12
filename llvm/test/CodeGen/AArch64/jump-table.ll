; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -filetype=obj | llvm-readobj -r | FileCheck %s -check-prefix=CHECK-ELF

define i32 @test_jumptable(i32 %in) {
; CHECK: test_jumptable

  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
  ]
; CHECK: adrp [[JTPAGE:x[0-9]+]], .LJTI0_0
; CHECK: add x[[JT:[0-9]+]], [[JTPAGE]], #:lo12:.LJTI0_0
; CHECK: ldr [[DEST:x[0-9]+]], [x[[JT]], {{x[0-9]+}}, lsl #3]
; CHECK: br [[DEST]]

def:
  ret i32 0

lbl1:
  ret i32 1

lbl2:
  ret i32 2

lbl3:
  ret i32 4

lbl4:
  ret i32 8

}

; CHECK: .rodata

; CHECK: .LJTI0_0:
; CHECK-NEXT: .xword
; CHECK-NEXT: .xword
; CHECK-NEXT: .xword
; CHECK-NEXT: .xword
; CHECK-NEXT: .xword

; ELF tests:

; First make sure we get a page/lo12 pair in .text to pick up the jump-table

; CHECK-ELF:      Relocations [
; CHECK-ELF:        Section ({{[0-9]+}}) .text {
; CHECK-ELF-NEXT:     0x{{[0-9,A-F]+}} R_AARCH64_ADR_PREL_PG_HI21 .rodata
; CHECK-ELF-NEXT:     0x{{[0-9,A-F]+}} R_AARCH64_ADD_ABS_LO12_NC .rodata
; CHECK-ELF:        }

; Also check the targets in .rodata are relocated
; CHECK-ELF:        Section ({{[0-9]+}}) .rodata {
; CHECK-ELF-NEXT:     0x{{[0-9,A-F]+}} R_AARCH64_ABS64 .text
; CHECK-ELF:        }
; CHECK-ELF:      ]
