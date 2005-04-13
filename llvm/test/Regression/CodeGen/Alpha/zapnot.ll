; Make sure this testcase codegens to the bic instruction
; RUN: llvm-as < %s | llc -march=alpha | grep 'zapnot'

implementation   ; Functions:

ushort %foo(long %y) {
entry:
        %tmp.1 = cast long %y to ushort         ; <ushort> [#uses=1]
        ret ushort %tmp.1
}

