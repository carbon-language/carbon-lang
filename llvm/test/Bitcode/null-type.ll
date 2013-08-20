; RUN: not llvm-dis < %s.bc 2>&1 | FileCheck %s
; PR8494

; CHECK: Invalid MODULE_CODE_FUNCTION record
