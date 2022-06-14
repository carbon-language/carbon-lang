; REQUIRES: x86
; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %S/Inputs/linker-script-symbols-ipo.ll -o %t2.o
; RUN: echo "bar = foo;" > %t.script

;; Check that without linkerscript bar is inlined.
; RUN: ld.lld %t1.o %t2.o -o %t3 -save-temps
; RUN: llvm-objdump -d %t3 | FileCheck %s --check-prefix=IPO
; IPO:      Disassembly of section .text:
; IPO:      <_start>:
; IPO-NEXT:   movl $1, %eax
; IPO-NEXT:   retq

;; Check that LTO does not do IPO for symbols assigned by script.
; RUN: ld.lld %t1.o %t2.o -o %t4 --script %t.script -save-temps
; RUN: llvm-objdump -d %t4 | FileCheck %s --check-prefix=NOIPO
; NOIPO:      Disassembly of section .text:
; NOIPO:      <foo>:
; NOIPO-NEXT:   movl $2, %eax
; NOIPO:      <_start>:
; NOIPO-NEXT:   jmp 0x201160 <foo>

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @bar() {
  ret i32 1
}

define i32 @foo() {
  ret i32 2
}
