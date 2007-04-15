; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6 | \
; RUN:   grep smmul | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep umull | wc -l | grep 1

int %smulhi(int %x, int %y) {
        %tmp = cast int %x to ulong             ; <ulong> [#uses=1]
        %tmp1 = cast int %y to ulong            ; <ulong> [#uses=1]
        %tmp2 = mul ulong %tmp1, %tmp           ; <ulong> [#uses=1]
        %tmp3 = shr ulong %tmp2, ubyte 32               ; <ulong> [#uses=1]
        %tmp3 = cast ulong %tmp3 to int         ; <int> [#uses=1]
        ret int %tmp3
}

int %umulhi(uint %x, uint %y) {
        %tmp = cast uint %x to ulong            ; <ulong> [#uses=1]
        %tmp1 = cast uint %y to ulong           ; <ulong> [#uses=1]
        %tmp2 = mul ulong %tmp1, %tmp           ; <ulong> [#uses=1]
        %tmp3 = shr ulong %tmp2, ubyte 32               ; <ulong> [#uses=1]
        %tmp3 = cast ulong %tmp3 to int         ; <int> [#uses=1]
        ret int %tmp3
}

