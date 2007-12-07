; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2

define void @test(<8 x i16>* %res, <8 x i16>* %A, <8 x i16>* %B) {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> < i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >
	store <8 x i16> %tmp3, <8 x i16>* %res
	ret void
}
