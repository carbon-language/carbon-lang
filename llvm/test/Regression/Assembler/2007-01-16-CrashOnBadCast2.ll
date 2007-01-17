; PR1117
; RUN: llvm-as < %s 2>&1 > /dev/null | \
; RUN:  grep "invalid cast opcode for cast from"

%X = constant i8* trunc (i64 0 to i8*)
