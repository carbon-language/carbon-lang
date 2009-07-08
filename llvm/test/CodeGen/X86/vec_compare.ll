; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah | grep pcmpgtd | count 2

define <4 x i32> @test(<4 x i32> %A, <4 x i32> %B) nounwind {
	%C = vicmp sgt <4 x i32> %A, %B
	ret <4 x i32> %C
}


define <4 x i32> @test2(<4 x i32> %A, <4 x i32> %B) nounwind {
	%C = icmp sgt <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

