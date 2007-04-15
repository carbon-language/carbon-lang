; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl | llvm-dis | \
; RUN:   not grep alloca
; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl | llvm-dis | \
; RUN:   grep {bitcast.*float.*i32}

implementation

int %test(float %X) {
        %X_addr = alloca float
        store float %X, float* %X_addr
        %X_addr = bitcast float* %X_addr to int*
        %tmp = load int* %X_addr
        ret int %tmp
}
