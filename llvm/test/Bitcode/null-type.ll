; RUN: not llvm-dis < %s.bc > /dev/null |& grep "Invalid MODULE_CODE_FUNCTION record"
; PR8494
