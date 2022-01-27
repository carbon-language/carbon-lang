; Generate summary sections and test gold handling.
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o
; Include a file with an empty module summary index, to ensure that the expected
; output files are created regardless, for a distributed build system.
; RUN: opt -module-summary %p/Inputs/thinlto_empty.ll -o %t3.o

; Ensure gold generates imports files if requested for distributed backends.
; RUN: rm -f %t3.o.imports %t3.o.thinlto.bc
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    --plugin-opt=thinlto-emit-imports-files \
; RUN:    -shared %t.o %t2.o %t3.o -o %t4

; The imports file for this module contains the bitcode file for
; Inputs/thinlto.ll
; RUN: cat %t.o.imports | count 1
; RUN: cat %t.o.imports | FileCheck %s --check-prefix=IMPORTS1
; IMPORTS1: test/tools/gold/X86/Output/thinlto_emit_imports.ll.tmp2.o

; The imports file for Input/thinlto.ll is empty as it does not import anything.
; RUN: cat %t2.o.imports | count 0

; The imports file for Input/thinlto_empty.ll is empty but should exist.
; RUN: cat %t3.o.imports | count 0

; The index file should be created even for the input with an empty summary.
; RUN: ls %t3.o.thinlto.bc

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
