; RUN: not llc -filetype=null %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".equiv var, __var"

@var = global i32 0
; CHECK: <unknown>:0: error: symbol 'var' is already defined
; CHECK: <unknown>:0: error: symbol 'var' is already defined
