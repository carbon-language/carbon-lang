; Test to make sure the thinlto-object-suffix-replace option is handled
; correctly.

; Generate bitcode file with summary, as well as a minimized bitcode without
; the debug metadata for the thin link.
; RUN: opt -thinlto-bc %s -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.o

; First perform the thin link on the normal bitcode file, and save the
; resulting index.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    -shared %t1.o -o %t3
; RUN: cp %t1.o.thinlto.bc %t1.o.thinlto.bc.orig

; Next perform the thin link on the minimized bitcode file, and compare dump
; of the resulting index to the above dump to ensure they are identical.
; RUN: rm -f %t1.o.thinlto.bc
; Make sure it isn't inadvertently using the regular bitcode file.
; RUN: rm -f %t1.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    --plugin-opt=thinlto-object-suffix-replace=".thinlink.bc;.o" \
; RUN:    -shared %t1.thinlink.bc -o %t3
; RUN: diff %t1.o.thinlto.bc.orig %t1.o.thinlto.bc

; If filename does not end with old suffix, no suffix change should occur,
; so ".thinlto.bc" will simply be appended to the input file name.
; RUN: rm -f %t1.thinlink.bc.thinlto.bc
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    --plugin-opt=thinlto-object-suffix-replace=".abc;.o" \
; RUN:    -shared %t1.thinlink.bc -o /dev/null
; RUN: ls %t1.thinlink.bc.thinlto.bc

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f() {
entry:
  ret void
}

!llvm.dbg.cu = !{}

!1 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!1}
