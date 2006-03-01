; RUN: llvm-as < %s | llc -march=x86 | grep lea
; RUN: llvm-as < %s | llc -march=x86 | not grep add

int %test(int %X, int %Y) {
	; Push the shl through the mul to allow an LEA to be formed, instead
        ; of using a shift and add separately.
        %tmp.2 = shl int %X, ubyte 1
        %tmp.3 = mul int %tmp.2, %Y
        %tmp.5 = add int %tmp.3, %Y
        ret int %tmp.5
}

