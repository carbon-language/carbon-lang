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
