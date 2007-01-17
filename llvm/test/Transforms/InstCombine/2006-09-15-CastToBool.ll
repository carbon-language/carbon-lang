; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep and
; PR913

int %test(int* %tmp1) {
        %tmp.i = load int* %tmp1          ; <int> [#uses=1]
        %tmp = cast int %tmp.i to uint            ; <uint> [#uses=1]
        %tmp2.ui = shr uint %tmp, ubyte 5          ; <uint> [#uses=1]
        %tmp2 = cast uint %tmp2.ui to int          ; <int> [#uses=1]
        %tmp3 = and int %tmp2, 1                ; <int> [#uses=1]
        %tmp3 = cast int %tmp3 to bool          ; <bool> [#uses=1]
        %tmp34 = cast bool %tmp3 to int         ; <int> [#uses=1]
        ret int %tmp34
}

