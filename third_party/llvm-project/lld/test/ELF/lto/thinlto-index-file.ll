; REQUIRES: x86

; Basic ThinLTO tests.
; RUN: opt -module-summary %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o
; RUN: opt -module-summary %p/Inputs/thinlto_empty.ll -o %t3.o

; Ensure lld writes linked files to linked objects file
; RUN: ld.lld --plugin-opt=thinlto-index-only=%t.idx -shared %t1.o %t2.o %t3.o -o /dev/null
; RUN: FileCheck %s < %t.idx
; CHECK: {{.*}}thinlto-index-file.ll.tmp1.o
; CHECK: {{.*}}thinlto-index-file.ll.tmp2.o
; CHECK: {{.*}}thinlto-index-file.ll.tmp3.o

; Check that this also works without the --plugin-opt= prefix.
; RUN: ld.lld --thinlto-index-only=%t.idx -shared %t1.o %t2.o %t3.o -o /dev/null
; RUN: FileCheck %s < %t.idx

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
