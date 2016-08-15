; This test ensures that we drop the preempted copy of @f from %t2.bc from its
; comdat after making it available_externally. If not we would get a
; verification error.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/linkonce_resolution_comdat.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=run %t1.bc %t2.bc -exported-symbol=f -exported-symbol=g

; RUN: llvm-nm -o - < %t1.bc.thinlto.o | FileCheck %s --check-prefix=NM1
; NM1: W f

; RUN: llvm-nm -o - < %t2.bc.thinlto.o | FileCheck %s --check-prefix=NM2
; f() would have been turned into available_externally since it is preempted,
; and inlined into g()
; NM2-NOT: f

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$c1 = comdat any

define linkonce_odr i32 @f(i8*) unnamed_addr comdat($c1) {
    ret i32 43
}
