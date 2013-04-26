; RUN: llc < %s -march=x86-64 -mcpu=x86-64 | FileCheck %s -check-prefix=SSE2
; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s -check-prefix=SSSE3
; RUN: llc < %s -march=x86-64 -mcpu=core-avx2 | FileCheck %s -check-prefix=AVX2

define <4 x i32> @test1(<4 x i32> %a) nounwind {
; SSE2: test1:
; SSE2: movdqa
; SSE2-NEXT: psrad $31
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret

; SSSE3: test1:
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2: test1:
; AVX2: vpabsd
; AVX2-NEXT: ret
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

; SSSE3: test2:
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2: test2:
; AVX2: vpabsd
; AVX2-NEXT: ret
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sge <4 x i32> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i32> %a, <4 x i32> %tmp1neg
        ret <4 x i32> %abs
}

define <8 x i16> @test3(<8 x i16> %a) nounwind {
; SSE2: test3:
; SSE2: movdqa
; SSE2-NEXT: psraw $15
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret

; SSSE3: test3:
; SSSE3: pabsw
; SSSE3-NEXT: ret

; AVX2: test3:
; AVX2: vpabsw
; AVX2-NEXT: ret
        %tmp1neg = sub <8 x i16> zeroinitializer, %a
        %b = icmp sgt <8 x i16> %a, zeroinitializer
        %abs = select <8 x i1> %b, <8 x i16> %a, <8 x i16> %tmp1neg
        ret <8 x i16> %abs
}

define <16 x i8> @test4(<16 x i8> %a) nounwind {
; SSE2: test4:
; SSE2: pxor
; SSE2-NEXT: pcmpgtb
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret

; SSSE3: test4:
; SSSE3: pabsb
; SSSE3-NEXT: ret

; AVX2: test4:
; AVX2: vpabsb
; AVX2-NEXT: ret
        %tmp1neg = sub <16 x i8> zeroinitializer, %a
        %b = icmp slt <16 x i8> %a, zeroinitializer
        %abs = select <16 x i1> %b, <16 x i8> %tmp1neg, <16 x i8> %a
        ret <16 x i8> %abs
}

define <4 x i32> @test5(<4 x i32> %a) nounwind {
; SSE2: test5:
; SSE2: movdqa
; SSE2-NEXT: psrad $31
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret

; SSSE3: test5:
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2: test5:
; AVX2: vpabsd
; AVX2-NEXT: ret
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sle <4 x i32> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i32> %tmp1neg, <4 x i32> %a
        ret <4 x i32> %abs
}

define <8 x i32> @test6(<8 x i32> %a) nounwind {
; SSSE3: test6:
; SSSE3: pabsd
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2: test6:
; AVX2: vpabsd %ymm
; AVX2-NEXT: ret
        %tmp1neg = sub <8 x i32> zeroinitializer, %a
        %b = icmp sgt <8 x i32> %a, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
        %abs = select <8 x i1> %b, <8 x i32> %a, <8 x i32> %tmp1neg
        ret <8 x i32> %abs
}

define <8 x i32> @test7(<8 x i32> %a) nounwind {
; SSSE3: test7:
; SSSE3: pabsd
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2: test7:
; AVX2: vpabsd %ymm
; AVX2-NEXT: ret
        %tmp1neg = sub <8 x i32> zeroinitializer, %a
        %b = icmp sge <8 x i32> %a, zeroinitializer
        %abs = select <8 x i1> %b, <8 x i32> %a, <8 x i32> %tmp1neg
        ret <8 x i32> %abs
}

define <16 x i16> @test8(<16 x i16> %a) nounwind {
; SSSE3: test8:
; SSSE3: pabsw
; SSSE3: pabsw
; SSSE3-NEXT: ret

; AVX2: test8:
; AVX2: vpabsw %ymm
; AVX2-NEXT: ret
        %tmp1neg = sub <16 x i16> zeroinitializer, %a
        %b = icmp sgt <16 x i16> %a, zeroinitializer
        %abs = select <16 x i1> %b, <16 x i16> %a, <16 x i16> %tmp1neg
        ret <16 x i16> %abs
}

define <32 x i8> @test9(<32 x i8> %a) nounwind {
; SSSE3: test9:
; SSSE3: pabsb
; SSSE3: pabsb
; SSSE3-NEXT: ret

; AVX2: test9:
; AVX2: vpabsb %ymm
; AVX2-NEXT: ret
        %tmp1neg = sub <32 x i8> zeroinitializer, %a
        %b = icmp slt <32 x i8> %a, zeroinitializer
        %abs = select <32 x i1> %b, <32 x i8> %tmp1neg, <32 x i8> %a
        ret <32 x i8> %abs
}

define <8 x i32> @test10(<8 x i32> %a) nounwind {
; SSSE3: test10:
; SSSE3: pabsd
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2: test10:
; AVX2: vpabsd %ymm
; AVX2-NEXT: ret
        %tmp1neg = sub <8 x i32> zeroinitializer, %a
        %b = icmp sle <8 x i32> %a, zeroinitializer
        %abs = select <8 x i1> %b, <8 x i32> %tmp1neg, <8 x i32> %a
        ret <8 x i32> %abs
}
