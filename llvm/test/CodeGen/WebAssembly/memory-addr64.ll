; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic memory operations assemble as expected with 64-bit addresses.

target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "wasm64-unknown-unknown"

declare i64 @llvm.wasm.page.size.i64() nounwind readnone
declare i64 @llvm.wasm.memory.size.i64() nounwind readnone
declare void @llvm.wasm.resize.memory.i64(i64) nounwind

; CHECK-LABEL: (func $page_size
; CHECK-NEXT: (result i64)
; CHECK-NEXT: (set_local @0 (page_size))
; CHECK-NEXT: (return @0)
define i64 @page_size() {
  %a = call i64 @llvm.wasm.page.size.i64()
  ret i64 %a
}

; CHECK-LABEL: (func $memory_size
; CHECK-NEXT: (result i64)
; CHECK-NEXT: (set_local @0 (memory_size))
; CHECK-NEXT: (return @0)
define i64 @memory_size() {
  %a = call i64 @llvm.wasm.memory.size.i64()
  ret i64 %a
}

; CHECK-LABEL: (func $resize_memory
; CHECK-NEXT: (param i64)
; CHECK: (resize_memory @0)
; CHECK-NEXT: (return)
define void @resize_memory(i64 %n) {
  call void @llvm.wasm.resize.memory.i64(i64 %n)
  ret void
}
