; Reassociation should apply to Add, Mul, And, Or, & Xor
;
; RUN: llvm-as < %s | opt -reassociate -constprop -instcombine -die | llvm-dis | not grep 12

int "test_mul"(int %arg) {
        %tmp1 = mul int 12, %arg
        %tmp2 = mul int %tmp1, 12
        ret int %tmp2
}

int "test_and"(int %arg) {
	%tmp1 = and int 14, %arg
	%tmp2 = and int %tmp1, 14
	ret int %tmp2
}

int "test_or"(int %arg) {
        %tmp1 = or int 14, %arg
        %tmp2 = or int %tmp1, 14
        ret int %tmp2
}

int "test_xor"(int %arg) {
        %tmp1 = xor int 12, %arg
        %tmp2 = xor int %tmp1, 12
        ret int %tmp2
}

