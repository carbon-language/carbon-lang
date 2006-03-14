; There should be exactly one shift and one add left.
; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | grep shl | wc -l | grep 1 &&
; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | grep add | wc -l | grep 1

int %test(int %X, int %Y) {
        %tmp.2 = shl int %X, ubyte 1            ; <int> [#uses=1]
        %tmp.6 = shl int %Y, ubyte 1            ; <int> [#uses=1]
        %tmp.4 = add int %tmp.6, %tmp.2         ; <int> [#uses=1]
        ret int %tmp.4
}

