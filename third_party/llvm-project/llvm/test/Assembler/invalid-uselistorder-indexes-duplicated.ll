; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: expected distinct uselistorder indexes in range [0, size)
@global = global i32 0
@alias1 = alias i32, i32* @global
@alias2 = alias i32, i32* @global
@alias3 = alias i32, i32* @global
uselistorder i32* @global, { 0, 0, 2 }
