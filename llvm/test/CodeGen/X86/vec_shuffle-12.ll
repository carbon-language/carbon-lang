; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep punpck
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pextrw | count 7
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pinsrw | count 7
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pshuf | count 2

define void @t1(<8 x i16>* %res, <8 x i16>* %A, <8 x i16>* %B) {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> < i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >
	store <8 x i16> %tmp3, <8 x i16>* %res
	ret void
}

define void @t2(<8 x i16>* %res, <8 x i16>* %A, <8 x i16>* %B) {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> < i32 8, i32 1, i32 2, i32 13, i32 4, i32 5, i32 6, i32 7 >
	store <8 x i16> %tmp3, <8 x i16>* %res
	ret void
}

define void @t3(<8 x i16>* %res, <8 x i16>* %A, <8 x i16>* %B) {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> < i32 8, i32 3, i32 2, i32 13, i32 7, i32 6, i32 5, i32 4 >
	store <8 x i16> %tmp3, <8 x i16>* %res
	ret void
}

define void @t4(<8 x i16>* %res, <8 x i16>* %A, <8 x i16>* %B) {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> < i32 8, i32 9, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >
	store <8 x i16> %tmp3, <8 x i16>* %res
	ret void
}
