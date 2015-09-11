; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: value only has one use
@global = global i32 0
@alias = alias i32, i32* @global
uselistorder i32* @global, { 1, 0 }
