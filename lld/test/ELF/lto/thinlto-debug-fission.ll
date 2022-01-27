; REQUIRES: x86

; RUN: opt %s -o %t1.o
; RUN: rm -rf %t.dir

; Test to ensure that --plugin-opt=dwo_dir=$DIR creates .dwo files under $DIR
; RUN: ld.lld --plugin-opt=dwo_dir=%t.dir -shared %t1.o -o /dev/null
; RUN: llvm-readobj -h %t.dir/0.dwo | FileCheck %s

; CHECK: Format: elf64-x86-64

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
