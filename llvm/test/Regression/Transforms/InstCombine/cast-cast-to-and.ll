; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep ubyte

int %test1(uint %X) {
        %Y = cast uint %X to ubyte ;; Turn into an AND
        %Z = cast ubyte %Y to int
        ret int %Z
}

