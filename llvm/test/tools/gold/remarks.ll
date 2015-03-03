; RUN: llvm-as %s -o %t.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    -plugin-opt=-pass-remarks=inline %t.o -o %t2.o 2>&1 | FileCheck %s

; CHECK: f inlined into _start
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f() {
  ret i32 0
}

define i32 @_start() {
  %call = call i32 @f()
  ret i32 %call
}
