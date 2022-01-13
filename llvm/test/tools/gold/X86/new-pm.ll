; Test plugin options new-pass-manager and debug-pass-manager
; RUN: opt -module-summary %s -o %t.o

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=new-pass-manager \
; RUN:     --plugin-opt=debug-pass-manager \
; RUN:     --plugin-opt=cache-dir=%t.cache \
; RUN:     -o %t2.o %t.o 2>&1 | FileCheck %s

; CHECK: Running pass:

;; --plugin-opt=debug-pass-manager is a no-op for the legacy pass manager.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=legacy-pass-manager --plugin-opt=debug-pass-manager \
; RUN:     -o /dev/null %t.o 2>&1 | count 0

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @globalfunc() #0 {
entry:
  ret void
}
