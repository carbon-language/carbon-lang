; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm

long %foo0(long %A, ulong %B) {
        %tmp = cast long %A to ulong            ; <ulong> [#uses=1]
        %tmp2 = shr ulong %B, ubyte 1           ; <ulong> [#uses=1]
        %tmp3 = sub ulong %tmp, %tmp2           ; <ulong> [#uses=1]
        %tmp3 = cast ulong %tmp3 to long                ; <long> [#uses=1]
        ret long %tmp3
}

