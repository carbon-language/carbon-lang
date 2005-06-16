; RUN: llvm-as < %s | opt -instcombine -disable-output

int %_Z13func_31585107li(int %l_39521025, int %l_59244666) {
	%shortcirc_val = select bool false, uint 1, uint 0		; <uint> [#uses=1]
	%tmp.8 = div uint 0, %shortcirc_val		; <uint> [#uses=1]
	%tmp.9 = seteq uint %tmp.8, 0		; <bool> [#uses=1]
	%retval = select bool %tmp.9, int %l_59244666, int -1621308501		; <int> [#uses=1]
	ret int %retval
}
