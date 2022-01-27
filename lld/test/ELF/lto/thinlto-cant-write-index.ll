; REQUIRES: x86

; Basic ThinLTO tests.
; RUN: opt -module-summary %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o

; Ensure lld generates error if unable to write to index files
; RUN: rm -f %t2.o.thinlto.bc
; RUN: touch %t2.o.thinlto.bc
; RUN: chmod u-w %t2.o.thinlto.bc
; RUN: not ld.lld --plugin-opt=thinlto-index-only -shared %t1.o %t2.o -o /dev/null 2>&1 | FileCheck -DMSG=%errc_EACCES %s
; RUN: chmod u+w %t2.o.thinlto.bc
; CHECK: cannot open {{.*}}2.o.thinlto.bc: [[MSG]]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
