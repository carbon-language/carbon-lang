; There should be exactly one shift and one add left.
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   opt -reassociate -instcombine | llvm-dis > %t  
; RUN: grep shl %t | wc -l | grep 1
; RUN: grep add %t | wc -l | grep 1

int %test(int %X, int %Y) {
        %tmp.2 = shl int %X, ubyte 1            ; <int> [#uses=1]
        %tmp.6 = shl int %Y, ubyte 1            ; <int> [#uses=1]
        %tmp.4 = add int %tmp.6, %tmp.2         ; <int> [#uses=1]
        ret int %tmp.4
}

