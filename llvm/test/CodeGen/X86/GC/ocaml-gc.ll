; RUN: llc < %s -mtriple=x86_64-linux-gnu | FileCheck %s

define i32 @main(i32 %x) nounwind gc "ocaml" {
; CHECK:        .text
; CHECK-NEXT:   .globl  caml_3C_stdin_3E___code_begin
; CHECK-NEXT: caml_3C_stdin_3E___code_begin:
; CHECK-NEXT:   .data
; CHECK-NEXT:   .globl  caml_3C_stdin_3E___data_begin
; CHECK-NEXT: caml_3C_stdin_3E___data_begin:

  %puts = tail call i32 @foo(i32 %x)
  ret i32 0

; CHECK:        .globl  caml_3C_stdin_3E___code_end
; CHECK-NEXT: caml_3C_stdin_3E___code_end:
; CHECK-NEXT:   .data
; CHECK-NEXT:   .globl  caml_3C_stdin_3E___data_end
; CHECK-NEXT: caml_3C_stdin_3E___data_end:
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .globl  caml_3C_stdin_3E___frametable
; CHECK-NEXT: caml_3C_stdin_3E___frametable:
; CHECK-NEXT:   .short  1
; CHECK-NEXT:   .align  8
; CHECK-NEXT:                # live roots for main
; CHECK-NEXT:   .quad   .Ltmp0
; CHECK-NEXT:   .short  8
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .align  8
}

declare i32 @foo(i32)
