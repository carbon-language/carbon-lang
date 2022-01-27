; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

; CHECK: input module has no datalayout

; This bitcode file has no datalayout.
; Check that we error out producing a reasonable diagnostic.
target triple = "x86_64-unknown-linux-gnu"

define void @_start() {
  ret void
}
