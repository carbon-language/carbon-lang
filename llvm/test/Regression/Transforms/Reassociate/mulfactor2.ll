; This should turn into one multiply and one add.

; RUN: llvm-as < %s | opt -instcombine -reassociate -instcombine | llvm-dis | grep mul | wc -l | grep 1 &&
; RUN: llvm-as < %s | opt -instcombine -reassociate -instcombine | llvm-dis | grep add | wc -l | grep 1
int %main(int %t) {
        %tmp.3 = mul int %t, 12         ; <int> [#uses=1]
        %tmp.4 = add int %tmp.3, 5              ; <int> [#uses=1]
        %tmp.6 = mul int %t, 6          ; <int> [#uses=1]
        %tmp.8 = mul int %tmp.4, 3              ; <int> [#uses=1]
        %tmp.9 = add int %tmp.8, %tmp.6         ; <int> [#uses=1]
        ret int %tmp.9
}

