; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: value has no uses
@global = global i32 0
uselistorder i32* @global, { 1, 0 }
