; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep {and i32 %Y, 8}

int %test1(ubyte %X) {
        %Y = cast ubyte %X to int
        %Z = and int %Y, 65544     ;; Prune this to and Y, 8
        ret int %Z
}

