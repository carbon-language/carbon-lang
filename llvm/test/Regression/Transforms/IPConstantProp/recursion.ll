; RUN: llvm-as < %s | opt -ipconstprop -deadargelim | llvm-dis | not grep %X

implementation

internal int %foo(int %X) {
        %Y = call int %foo( int %X )
        %Z = add int %Y, 1
        ret int %Z
}

void %bar() {
        call int %foo( int 17 )         ; <int>:0 [#uses=0]
        ret void
}
