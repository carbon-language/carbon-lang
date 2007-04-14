; Tests to make sure elimination of casts is working correctly
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   notcast {} {%c1.*}

long %test_sext_zext(short %A) {
    %c1 = zext short %A to uint
    %c2 = sext uint %c1 to long
    ret long %c2
}
