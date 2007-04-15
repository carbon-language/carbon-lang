; PR1137
; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f
; RUN: llvm-upgrade < %s | grep {tmp = alloca} | wc -l | grep 1
;
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"

implementation   ; Functions:

void %main() {
entry:
    %tmp = alloca uint, align 4             ; <uint*> [#uses=1]
    %tmp = alloca int, align 4              ; <int*> [#uses=1]
    "alloca point" = cast int 0 to int      ; <int> [#uses=0]
    store uint 1, uint* %tmp
    store int 2, int* %tmp
    ret void
}
