; This test should not contain a sign extend
; RUN: llvm-as < %s | llc -march=ppc32 | not grep extsb 

int %test(uint %mode.0.i.0) {
        %tmp.79 = cast uint %mode.0.i.0 to sbyte        ; <sbyte> [#uses=1]
        %tmp.80 = cast sbyte %tmp.79 to int             ; <int> [#uses=1]
        %tmp.81 = shl int %tmp.80, ubyte 24             ; <int> [#uses=1]
        ret int %tmp.81
}

int %test2(uint %mode.0.i.0) {
        %tmp.79 = cast uint %mode.0.i.0 to sbyte        ; <sbyte> [#uses=1]
        %tmp.80 = cast sbyte %tmp.79 to int             ; <int> [#uses=1]
        %tmp.81 = shl int %tmp.80, ubyte 16             ; <int> [#uses=1]
        %tmp.82 = and int %tmp.81, 16711680
        ret int %tmp.82
}
