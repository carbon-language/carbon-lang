; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic memory operations assemble as expected with 32-bit addresses.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @llvm.wasm.page.size.i32() nounwind readnone
declare i32 @llvm.wasm.memory.size.i32() nounwind readnone
declare void @llvm.wasm.resize.memory.i32(i32) nounwind

; CHECK-LABEL: (func $page_size
; CHECK-NEXT: (result i32)
; CHECK-NEXT: (setlocal @0 (page_size))
; CHECK-NEXT: (return @0)
define i32 @page_size() {
  %a = call i32 @llvm.wasm.page.size.i32()
  ret i32 %a
}

; CHECK-LABEL: (func $memory_size
; CHECK-NEXT: (result i32)
; CHECK-NEXT: (setlocal @0 (memory_size))
; CHECK-NEXT: (return @0)
define i32 @memory_size() {
  %a = call i32 @llvm.wasm.memory.size.i32()
  ret i32 %a
}

; CHECK-LABEL: (func $resize_memory
; CHECK-NEXT: (param i32)
; CHECK: (resize_memory @0)
; CHECK-NEXT: (return)
define void @resize_memory(i32 %n) {
  call void @llvm.wasm.resize.memory.i32(i32 %n)
  ret void
}
