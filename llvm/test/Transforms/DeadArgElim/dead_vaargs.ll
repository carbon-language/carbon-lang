; RUN: llvm-upgrade < %s | llvm-as | opt -deadargelim | llvm-dis | not grep 47 
; RUN: llvm-upgrade < %s | llvm-as | opt -deadargelim | llvm-dis | not grep 1.0

int %bar(int %A) {
        %tmp4 = tail call int (int, ...)* %foo( int %A, int %A, int %A, int %A, ulong 47, double 1.000000e+00 )
        ret int %tmp4
}

internal int %foo(int %X, ...) {
        ret int %X
}

