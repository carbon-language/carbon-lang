; RUN: llc -mtriple=x86_64-apple-darwin -mcpu=corei7 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu -mcpu=corei7 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-windows-gnu -mcpu=corei7 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-windows-msvc -mcpu=corei7 < %s | FileCheck %s

define webkit_jscc i32 @simple_jscall(i32 %a, i32 %b, i32 %c) {
  %ab = add i32 %a, %b
  %abc = add i32 %ab, %c
  ret i32 %abc
}

; 32-bit integers are only aligned to 4 bytes, even on x64. They are *not*
; promoted to i64.

; CHECK: simple_jscall:
; CHECK: addl 8(%rsp), %eax
; CHECK-NEXT: addl 12(%rsp), %eax
; CHECK-NEXT: retq
