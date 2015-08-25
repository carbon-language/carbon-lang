; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic memory operations assemble as expected with 32-bit addresses.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @llvm.wasm.page.size.i32() nounwind readnone

; CHECK-LABEL: (func $page_size
; CHECK-NEXT: (result i32)
; CHECK-NEXT: (setlocal @0 (page_size))
; CHECK-NEXT: (return @0)
define i32 @page_size() {
  %a = call i32 @llvm.wasm.page.size.i32()
  ret i32 %a
}
