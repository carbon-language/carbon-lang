; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

@global = global i32 0

@alias = internal hidden alias i32, i32* @global
; CHECK: symbol with local linkage must have default visibility
