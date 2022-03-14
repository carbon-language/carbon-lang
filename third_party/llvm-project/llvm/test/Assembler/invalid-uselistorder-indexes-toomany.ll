; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: wrong number of indexes, expected 2
@global = global i32 0
@alias1 = alias i32, i32* @global
@alias2 = alias i32, i32* @global
uselistorder i32* @global, { 1, 0, 2 }
