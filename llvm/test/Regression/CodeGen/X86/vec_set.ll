; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep punpckl | wc -l | grep 7

void %test(<8 x short>* %b, short %a0, short %a1, short %a2, short %a3, short %a4, short %a5, short %a6, short %a7) {
	%tmp = insertelement <8 x short> zeroinitializer, short %a0, uint 0
	%tmp2 = insertelement <8 x short> %tmp, short %a1, uint 1
	%tmp4 = insertelement <8 x short> %tmp2, short %a2, uint 2
	%tmp6 = insertelement <8 x short> %tmp4, short %a3, uint 3
	%tmp8 = insertelement <8 x short> %tmp6, short %a4, uint 4
	%tmp10 = insertelement <8 x short> %tmp8, short %a5, uint 5
	%tmp12 = insertelement <8 x short> %tmp10, short %a6, uint 6
	%tmp14 = insertelement <8 x short> %tmp12, short %a7, uint 7
	store <8 x short> %tmp14, <8 x short>* %b
	ret void
}
