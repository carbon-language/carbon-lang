; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep {vtst\\.i8} %t | count 2
; RUN: grep {vtst\\.i16} %t | count 2
; RUN: grep {vtst\\.i32} %t | count 2

define <8 x i8> @vtsti8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = and <8 x i8> %tmp1, %tmp2
	%tmp4 = icmp ne <8 x i8> %tmp3, zeroinitializer
        %tmp5 = sext <8 x i1> %tmp4 to <8 x i8>
	ret <8 x i8> %tmp5
}

define <4 x i16> @vtsti16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = and <4 x i16> %tmp1, %tmp2
	%tmp4 = icmp ne <4 x i16> %tmp3, zeroinitializer
        %tmp5 = sext <4 x i1> %tmp4 to <4 x i16>
	ret <4 x i16> %tmp5
}

define <2 x i32> @vtsti32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = and <2 x i32> %tmp1, %tmp2
	%tmp4 = icmp ne <2 x i32> %tmp3, zeroinitializer
        %tmp5 = sext <2 x i1> %tmp4 to <2 x i32>
	ret <2 x i32> %tmp5
}

define <16 x i8> @vtstQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = and <16 x i8> %tmp1, %tmp2
	%tmp4 = icmp ne <16 x i8> %tmp3, zeroinitializer
        %tmp5 = sext <16 x i1> %tmp4 to <16 x i8>
	ret <16 x i8> %tmp5
}

define <8 x i16> @vtstQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = and <8 x i16> %tmp1, %tmp2
	%tmp4 = icmp ne <8 x i16> %tmp3, zeroinitializer
        %tmp5 = sext <8 x i1> %tmp4 to <8 x i16>
	ret <8 x i16> %tmp5
}

define <4 x i32> @vtstQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = and <4 x i32> %tmp1, %tmp2
	%tmp4 = icmp ne <4 x i32> %tmp3, zeroinitializer
        %tmp5 = sext <4 x i1> %tmp4 to <4 x i32>
	ret <4 x i32> %tmp5
}
