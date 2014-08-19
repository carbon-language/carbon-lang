; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: '@global' defined with type 'i32*'
@global = global i32 0
uselistorder i31* @global, { 1, 0 }
