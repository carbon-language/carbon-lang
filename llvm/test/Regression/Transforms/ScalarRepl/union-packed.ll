; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl | llvm-dis | not grep alloca &&
; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl | llvm-dis | grep 'bitcast'

; XFAIL: *

target endian = little

<4 x int> %test(<4 x float> %X) {
        %X_addr = alloca <4 x float>
        store <4 x float> %X, <4 x float>* %X_addr
        %X_addr = bitcast <4 x float>* %X_addr to <4 x int>*
        %tmp = load <4 x int>* %X_addr
        ret <4 x int> %tmp
}
