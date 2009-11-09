; RUN: llc < %s -march=arm -mattr=+neon | grep vldr.64 | count 4
; RUN: llc < %s -march=arm -mattr=+neon | grep vstr.64
; RUN: llc < %s -march=arm -mattr=+neon | grep vmov

define void @t1(<2 x i32>* %r, <4 x i16>* %a, <4 x i16>* %b) nounwind {
entry:
	%0 = load <4 x i16>* %a, align 8		; <<4 x i16>> [#uses=1]
	%1 = load <4 x i16>* %b, align 8		; <<4 x i16>> [#uses=1]
	%2 = add <4 x i16> %0, %1		; <<4 x i16>> [#uses=1]
	%3 = bitcast <4 x i16> %2 to <2 x i32>		; <<2 x i32>> [#uses=1]
	store <2 x i32> %3, <2 x i32>* %r, align 8
	ret void
}

define <2 x i32> @t2(<4 x i16>* %a, <4 x i16>* %b) nounwind readonly {
entry:
	%0 = load <4 x i16>* %a, align 8		; <<4 x i16>> [#uses=1]
	%1 = load <4 x i16>* %b, align 8		; <<4 x i16>> [#uses=1]
	%2 = sub <4 x i16> %0, %1		; <<4 x i16>> [#uses=1]
	%3 = bitcast <4 x i16> %2 to <2 x i32>		; <<2 x i32>> [#uses=1]
	ret <2 x i32> %3
}
