; RUN: llc < %s

define <8 x i32> @test1(<8 x i32>* %ptr)
{
	%1 = load <8 x i32>* %ptr, align 32
	%2 = and <8 x i32> %1, <i32 0, i32 0, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 -1>
	ret <8 x i32> %2;
}
