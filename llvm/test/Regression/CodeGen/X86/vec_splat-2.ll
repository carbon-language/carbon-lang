; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pshufd | wc -l | grep 1

void %test(<2 x long>* %P, sbyte %x) {
	%tmp = insertelement <16 x sbyte> zeroinitializer, sbyte %x, uint 0		; <<16 x sbyte>> [#uses=1]
	%tmp36 = insertelement <16 x sbyte> %tmp, sbyte %x, uint 1
	%tmp38 = insertelement <16 x sbyte> %tmp36, sbyte %x, uint 2
	%tmp40 = insertelement <16 x sbyte> %tmp38, sbyte %x, uint 3
	%tmp42 = insertelement <16 x sbyte> %tmp40, sbyte %x, uint 4
	%tmp44 = insertelement <16 x sbyte> %tmp42, sbyte %x, uint 5
	%tmp46 = insertelement <16 x sbyte> %tmp44, sbyte %x, uint 6
	%tmp48 = insertelement <16 x sbyte> %tmp46, sbyte %x, uint 7
	%tmp50 = insertelement <16 x sbyte> %tmp48, sbyte %x, uint 8
	%tmp52 = insertelement <16 x sbyte> %tmp50, sbyte %x, uint 9
	%tmp54 = insertelement <16 x sbyte> %tmp52, sbyte %x, uint 10
	%tmp56 = insertelement <16 x sbyte> %tmp54, sbyte %x, uint 11
	%tmp58 = insertelement <16 x sbyte> %tmp56, sbyte %x, uint 12
	%tmp60 = insertelement <16 x sbyte> %tmp58, sbyte %x, uint 13
	%tmp62 = insertelement <16 x sbyte> %tmp60, sbyte %x, uint 14
	%tmp64 = insertelement <16 x sbyte> %tmp62, sbyte %x, uint 15
	%tmp68 = load <2 x long>* %P
	%tmp71 = cast <2 x long> %tmp68 to <16 x sbyte>
	%tmp73 = add <16 x sbyte> %tmp71, %tmp64
	%tmp73 = cast <16 x sbyte> %tmp73 to <2 x long>
	store <2 x long> %tmp73, <2 x long>* %P
	ret void
}
