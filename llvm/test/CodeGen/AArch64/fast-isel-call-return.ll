; RUN: llc -fast-isel -fast-isel-abort -verify-machineinstrs < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

define i8* @test_call_return_type(i64 %size) {
entry:
; CHECK: bl xmalloc
  %0 = call noalias i8* @xmalloc(i64 undef)
  ret i8* %0
}

declare noalias i8* @xmalloc(i64)
