; The reassociate pass is not preserving dominance properties correctly
;
; RUN: llvm-upgrade < %s | llvm-as | opt -reassociate

int %compute_dist(int %i, int %j) {
        %reg119 = sub int %j, %i
        ret int %reg119
}


