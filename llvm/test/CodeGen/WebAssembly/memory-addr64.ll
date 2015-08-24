; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic memory operations assemble as expected with 64-bit addresses.

target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "wasm64-unknown-unknown"

declare i64 @llvm.wasm.page.size.i64() nounwind readnone

; CHECK-LABEL: page_size:
; CHECK-NEXT: (setlocal @0 (page_size))
; CHECK-NEXT: (return @0)
define i64 @page_size() {
  %a = call i64 @llvm.wasm.page.size.i64()
  ret i64 %a
}
