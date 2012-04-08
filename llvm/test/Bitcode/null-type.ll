; RUN: not llvm-dis < %s.bc > /dev/null |& FileCheck %s
; PR8494

; CHECK: Invalid MODULE_CODE_FUNCTION record
