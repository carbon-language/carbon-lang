; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pinsrw | wc -l | grep 2

<2 x long> %test(short %a) {
entry:
	%tmp10 = insertelement <8 x short> zeroinitializer, short %a, uint 3		; <<8 x short>> [#uses=1]
	%tmp12 = insertelement <8 x short> %tmp10, short 0, uint 4		; <<8 x short>> [#uses=1]
	%tmp14 = insertelement <8 x short> %tmp12, short 0, uint 5		; <<8 x short>> [#uses=1]
	%tmp16 = insertelement <8 x short> %tmp14, short 0, uint 6		; <<8 x short>> [#uses=1]
	%tmp18 = insertelement <8 x short> %tmp16, short 0, uint 7		; <<8 x short>> [#uses=1]
	%tmp19 = cast <8 x short> %tmp18 to <2 x long>		; <<2 x long>> [#uses=1]
	ret <2 x long> %tmp19
}

<2 x long> %test(sbyte %a) {
entry:
	%tmp24 = insertelement <16 x sbyte> zeroinitializer, sbyte %a, uint 10
	%tmp26 = insertelement <16 x sbyte> %tmp24, sbyte 0, uint 11
	%tmp28 = insertelement <16 x sbyte> %tmp26, sbyte 0, uint 12
	%tmp30 = insertelement <16 x sbyte> %tmp28, sbyte 0, uint 13
	%tmp32 = insertelement <16 x sbyte> %tmp30, sbyte 0, uint 14
	%tmp34 = insertelement <16 x sbyte> %tmp32, sbyte 0, uint 15
	%tmp35 = cast <16 x sbyte> %tmp34 to <2 x long>
	ret <2 x long> %tmp35
}
