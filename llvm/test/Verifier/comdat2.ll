; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

$v = comdat any
@v = private global i32 0, comdat $v
; CHECK: comdat global value has local linkage
