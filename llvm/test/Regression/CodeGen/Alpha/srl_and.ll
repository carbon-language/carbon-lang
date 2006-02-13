; Make sure this testcase codegens to the zapnot instruction
; RUN: llvm-as < %s | llc -march=alpha | grep 'zapnot'

ulong %foo(ulong %y) {
entry:
        %tmp = shr ulong %y, ubyte 3            ; <ulong> [#uses=1]
        %tmp2 = and ulong %tmp, 8191            ; <ulong> [#uses=1]
        ret ulong %tmp2
}

