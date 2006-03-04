; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | grep mul | wc -l | grep 2

; This should have exactly 2 multiplies when we're done.

int %f(int %a, int %b) {
        %tmp.2 = mul int %a, %a
        %tmp.5 = shl int %a, ubyte 1
        %tmp.6 = mul int %tmp.5, %b
        %tmp.10 = mul int %b, %b
        %tmp.7 = add int %tmp.6, %tmp.2
        %tmp.11 = add int %tmp.7, %tmp.10
        ret int %tmp.11
}

