; With reassociation, constant folding can eliminate the +/- 30 constants.
;
; RUN: as < %s | opt -reassociate -constprop -instcombine -die | dis | not grep 30

int "test"(int %reg109, int %reg1111) {
        %reg115 = add int %reg109, -30           ; <int> [#uses=1]
        %reg116 = add int %reg115, %reg1111             ; <int> [#uses=1]
        %reg117 = add int %reg116, 30           ; <int> [#uses=1]
        ret int %reg117
}
