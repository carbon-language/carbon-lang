; REQUIRES: x86
; RUN: rm -f %t.so %t.so.lto.bc
; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/save-temps.ll -o %t2.o
; RUN: ld.lld -shared -m elf_x86_64 %t.o %t2.o -o %t.so -save-temps
; RUN: llvm-nm %t.so | FileCheck %s
; RUN: llvm-nm %t.so.lto.bc | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

; CHECK-DAG: T bar
; CHECK-DAG: T foo
