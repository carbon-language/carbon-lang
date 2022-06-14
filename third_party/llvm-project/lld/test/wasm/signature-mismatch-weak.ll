; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/weak-symbol1.s -o %t.weak.o
; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/strong-symbol.s -o %t.strong.o
; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld -o %t.wasm %t.o %t.strong.o %t.weak.o 2>&1 | FileCheck %s

target triple = "wasm32-unknown-unknown"

declare i32 @weakFn() local_unnamed_addr

define void @_start() local_unnamed_addr {
entry:
  %call = call i32 @weakFn()
  ret void
}

; CHECK: warning: function signature mismatch: weakFn
; CHECK-NEXT: >>> defined as () -> i32 in {{.*}}signature-mismatch-weak.ll.tmp.o
; CHECK-NEXT: >>> defined as () -> i64 in {{.*}}signature-mismatch-weak.ll.tmp.strong.o
