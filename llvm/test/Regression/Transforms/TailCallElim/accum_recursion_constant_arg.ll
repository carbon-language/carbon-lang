; This is a more aggressive form of accumulator recursion insertion, which 
; requires noticing that X doesn't change as we perform the tailcall.  Thanks
; go out to the anonymous users of the demo script for "suggesting" 
; optimizations that should be done.  :)

; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | not grep call

int %mul(int %x, int %y) {
entry:
        %tmp.1 = seteq int %y, 0
        br bool %tmp.1, label %return, label %endif

endif:
        %tmp.8 = add int %y, -1
        %tmp.5 = call int %mul( int %x, int %tmp.8 )
        %tmp.9 = add int %tmp.5, %x
        ret int %tmp.9

return:
        ret int %x
}

