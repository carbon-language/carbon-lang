; Test mixed-mode LTO (mix of regular and thin LTO objects)
; RUN: opt %s -o %t.o
; RUN: opt -module-summary %p/Inputs/mixed_lto.ll -o %t2.o

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     -shared \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=-import-instr-limit=0 \
; RUN:     -o %t3.o %t2.o %t.o
; RUN: llvm-nm %t3.o | FileCheck %s

; CHECK-DAG: T main
; CHECK-DAG: T g

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
define i32 @g() {
  ret i32 0
}
