; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | not grep call

int %factorial(int %x) {
entry:
        %tmp.1 = setgt int %x, 0
        br bool %tmp.1, label %then, label %else

then:
        %tmp.6 = add int %x, -1
        %tmp.4 = call int %factorial( int %tmp.6 )
        %tmp.7 = mul int %tmp.4, %x
        ret int %tmp.7

else:
        ret int 1
}

