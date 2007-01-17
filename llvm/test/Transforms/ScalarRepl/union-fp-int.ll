; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl | llvm-dis | not grep alloca &&
; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl | llvm-dis | grep 'bitcast.*float.*i32'

int %test(float %X) {
        %X_addr = alloca float
        store float %X, float* %X_addr
        %X_addr = bitcast float* %X_addr to int*
        %tmp = load int* %X_addr
        ret int %tmp
}
