; PR1117
; RUN: llvm-as < %s 2>&1 > /dev/null | \
; RUN:  grep "invalid cast opcode for cast from"

define i8* %nada(i64 %X) {
    %result = trunc i64 %X to i8*
    ret i8* %result
}
