; RUN: llc < %s -march=x86-64 -mcpu=x86-64 | FileCheck %s -check-prefix=SSE2

define <4 x i32> @test1(<4 x i32> %a) nounwind {
; SSE2: test1:
; SSE2: movdqa
; SSE2-NEXT: psrad $31
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sgt <4 x i32> %a, <i32 -1, i32 -1, i32 -1, i32 -1>
        %abs = select <4 x i1> %b, <4 x i32> %a, <4 x i32> %tmp1neg
        ret <4 x i32> %abs
}

define <4 x i32> @test2(<4 x i32> %a) nounwind {
; SSE2: test2:
; SSE2: movdqa
; SSE2-NEXT: psrad $31
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sge <4 x i32> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i32> %a, <4 x i32> %tmp1neg
        ret <4 x i32> %abs
}

define <4 x i32> @test3(<4 x i32> %a) nounwind {
; SSE2: test3:
; SSE2: movdqa
; SSE2-NEXT: psrad $31
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sgt <4 x i32> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i32> %a, <4 x i32> %tmp1neg
        ret <4 x i32> %abs
}

define <4 x i32> @test4(<4 x i32> %a) nounwind {
; SSE2: test4:
; SSE2: movdqa
; SSE2-NEXT: psrad $31
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp slt <4 x i32> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i32> %tmp1neg, <4 x i32> %a
        ret <4 x i32> %abs
}

define <4 x i32> @test5(<4 x i32> %a) nounwind {
; SSE2: test5:
; SSE2: movdqa
; SSE2-NEXT: psrad $31
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sle <4 x i32> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i32> %tmp1neg, <4 x i32> %a
        ret <4 x i32> %abs
}
