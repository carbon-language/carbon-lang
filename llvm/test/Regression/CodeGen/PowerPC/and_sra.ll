; Neither of these functions should contain algebraic right shifts
; RUN: llvm-as < %s | llc -march=ppc32 | not grep srawi 

int %test1(uint %mode.0.i.0) {
        %tmp.79 = cast uint %mode.0.i.0 to int        ; <sbyte> [#uses=1]
        %tmp.80 = shr int %tmp.79, ubyte 15           ; <int> [#uses=1]
        %tmp.81 = and int %tmp.80, 24             ; <int> [#uses=1]
        ret int %tmp.81
}

int %test2(uint %mode.0.i.0) {
        %tmp.79 = cast uint %mode.0.i.0 to int        ; <sbyte> [#uses=1]
        %tmp.80 = shr int %tmp.79, ubyte 15           ; <int> [#uses=1]
        %tmp.81 = shr uint %mode.0.i.0, ubyte 16
        %tmp.82 = cast uint %tmp.81 to int
        %tmp.83 = and int %tmp.80, %tmp.82             ; <int> [#uses=1]
        ret int %tmp.83
}

uint %test3(int %specbits.6.1) {
        %tmp.2540 = shr int %specbits.6.1, ubyte 11             ; <int> [#uses=1]
        %tmp.2541 = cast int %tmp.2540 to uint          ; <uint> [#uses=1]
        %tmp.2542 = shl uint %tmp.2541, ubyte 13                ; <uint> [#uses=1]
        %tmp.2543 = and uint %tmp.2542, 8192            ; <uint> [#uses=1]
        ret uint %tmp.2543
}
