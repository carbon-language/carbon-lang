; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah | grep pcmpgtd

define <4 x i32> @test(<4 x i32> %A, <4 x i32> %B) nounwind {
	%C = vicmp sgt <4 x i32> %A, %B
	ret <4 x i32> %C
}

