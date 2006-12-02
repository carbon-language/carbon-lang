; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep rlwimi

void %test(short %div.0.i.i.i.i, int %L_num.0.i.i.i.i, int %tmp1.i.i206.i.i, short* %P) {
        %X = shl short %div.0.i.i.i.i, ubyte 1          ; <short> [#uses=1]
        %tmp28.i.i.i.i = shl int %L_num.0.i.i.i.i, ubyte 1              ; <int> [#uses=2]
        %tmp31.i.i.i.i = setlt int %tmp28.i.i.i.i, %tmp1.i.i206.i.i             ; <bool> [#uses=2]

        %tmp31.i.i.i.i = cast bool %tmp31.i.i.i.i to short              ; <short> [#uses=1]
        %tmp371.i.i.i.i1 = or short %tmp31.i.i.i.i, %X          ; <short> [#uses=1]
        %div.0.be.i.i.i.i = xor short %tmp371.i.i.i.i1, 1               ; <short> [#uses=1]
        store short %div.0.be.i.i.i.i, short* %P
        ret void
}

