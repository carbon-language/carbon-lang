; With reassociation, constant folding can eliminate the +/- 30 constants.
;
; RUN: if as < %s | opt -reassociate -constprop -instcombine -die | dis | grep 30
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test"(int %reg109, int %reg1111) {
        %reg115 = add int %reg109, -30           ; <int> [#uses=1]
        %reg116 = add int %reg115, %reg1111             ; <int> [#uses=1]
        %reg117 = add int %reg116, 30           ; <int> [#uses=1]
        ret int %reg117
}
