; RUN: not llvm-as < %s 2>&1 | FileCheck %s

global i64 u0x0p001
; CHECK: expected value token
