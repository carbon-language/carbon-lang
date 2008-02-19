; RUN: not llvm-as %s |& grep {is invalid or}
; PR2060

define i8* @foo() {
       ret i8* 0
}
