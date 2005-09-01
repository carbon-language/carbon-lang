; All of these ands and shifts should be folded into rlwimi's
; RUN: llvm-as < mulhs.ll | llc -march=ppc32 | not grep mulhwu && 
; RUN: llvm-as < mulhs.ll | llc -march=ppc32 | not grep srawi && 
; RUN: llvm-as < mulhs.ll | llc -march=ppc32 | not grep add && 
; RUN: llvm-as < mulhs.ll | llc -march=ppc32 | grep mulhw | wc -l | grep 1

implementation   ; Functions:

int %mulhs(int %a, int %b) {
entry:
        %tmp.1 = cast int %a to ulong           ; <ulong> [#uses=1]
        %tmp.3 = cast int %b to ulong           ; <ulong> [#uses=1]
        %tmp.4 = mul ulong %tmp.3, %tmp.1       ; <ulong> [#uses=1]
        %tmp.6 = shr ulong %tmp.4, ubyte 32     ; <ulong> [#uses=1]
        %tmp.7 = cast ulong %tmp.6 to int       ; <int> [#uses=1]
        ret int %tmp.7
}
