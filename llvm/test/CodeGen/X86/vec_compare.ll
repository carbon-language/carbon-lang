; RUN: llc < %s -march=x86 -mcpu=yonah -mtriple=i386-apple-darwin | FileCheck %s


define <4 x i32> @test1(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: test1:
; CHECK: pcmpgtd
; CHECK: ret

	%C = icmp sgt <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <4 x i32> @test2(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: test2:
; CHECK: pcmp
; CHECK: pcmp
; CHECK: pxor
; CHECK: ret
	%C = icmp sge <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <4 x i32> @test3(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: test3:
; CHECK: pcmpgtd
; CHECK: movdqa
; CHECK: ret
	%C = icmp slt <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <4 x i32> @test4(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: test4:
; CHECK: movdqa
; CHECK: pcmpgtd
; CHECK: ret
	%C = icmp ugt <4 x i32> %A, %B
        %D = sext <4 x i1> %C to <4 x i32>
	ret <4 x i32> %D
}

define <2 x i64> @test5(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK: test5:
; CHECK: pcmpeqd
; CHECK: pshufd $-79
; CHECK: pand
; CHECK: ret
	%C = icmp eq <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}

define <2 x i64> @test6(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK: test6:
; CHECK: pcmpeqd
; CHECK: pshufd $-79
; CHECK: pand
; CHECK: pcmpeqd
; CHECK: pxor
; CHECK: ret
	%C = icmp ne <2 x i64> %A, %B
	%D = sext <2 x i1> %C to <2 x i64>
	ret <2 x i64> %D
}
