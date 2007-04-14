; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep and.*32
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    not grep or.*153
; PR1014

int %test(int %tmp1) {
        %ovm = and int %tmp1, 32                ; <int> [#uses=1]
        %ov3 = add int %ovm, 145                ; <int> [#uses=2]
        %ov110 = xor int %ov3, 153
        ret int %ov110
}

