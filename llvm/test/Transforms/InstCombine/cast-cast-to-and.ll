; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   not grep ubyte

int %test1(uint %X) {
        %Y = cast uint %X to ubyte ;; Turn into an AND
        %Z = cast ubyte %Y to int
        ret int %Z
}

