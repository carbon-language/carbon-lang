; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep and

sbyte %test21(sbyte %A) {
        %C = shr sbyte %A, ubyte 7   ;; sign extend
        %D = and sbyte %C, 1         ;; chop off sign
        ret sbyte %D
}

