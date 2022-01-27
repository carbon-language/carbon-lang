; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -o %t.so -save-temps --lto-aa-pipeline=basic-aa \
; RUN: --lto-newpm-passes=ipsccp -shared
; RUN: ld.lld %t.o -o %t2.so -save-temps --lto-newpm-passes=loweratomic -shared
; RUN: llvm-dis %t.so.0.4.opt.bc -o - | FileCheck %s
; RUN: llvm-dis %t2.so.0.4.opt.bc -o - | FileCheck %s --check-prefix=ATOMIC

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @barrier() {
  fence seq_cst
  ret void
}

; IPSCCP won't remove the fence.
; CHECK: define void @barrier() {
; CHECK-NEXT: fence seq_cst
; CHECK-NEXT: ret void

; LowerAtomic will remove the fence.
; ATOMIC: define void @barrier() {
; ATOMIC-NEXT: ret void

; Check that invalid passes are rejected gracefully.
; RUN: not --crash ld.lld -m elf_x86_64 %t.o -o /dev/null \
; RUN:   --lto-newpm-passes=iamnotapass -shared 2>&1 | \
; RUN:   FileCheck %s --check-prefix=INVALID
; INVALID: unable to parse pass pipeline description 'iamnotapass': unknown pass name 'iamnotapass'

; Check that invalid AA pipelines are rejected gracefully.
; RUN: not --crash ld.lld -m elf_x86_64 %t.o -o /dev/null \
; RUN:   --lto-newpm-passes=globaldce --lto-aa-pipeline=patatino \
; RUN:   -shared 2>&1 | \
; RUN:   FileCheck %s --check-prefix=INVALIDAA
; INVALIDAA: unknown alias analysis name 'patatino'
