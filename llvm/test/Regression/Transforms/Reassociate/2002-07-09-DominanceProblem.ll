; The reassociate pass is not preserving dominance properties correctly
;
; RUN: as < %s | opt -reassociate -verify

int %compute_dist(int %i, int %j) {
        %reg119 = sub int %j, %i
        ret int %reg119
}


