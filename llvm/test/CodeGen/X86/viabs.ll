; RUN: llc < %s -march=x86-64 -mattr=sse2    | FileCheck %s -check-prefix=SSE2
; RUN: llc < %s -march=x86-64 -mattr=ssse3   | FileCheck %s -check-prefix=SSSE3
; RUN: llc < %s -march=x86-64 -mattr=avx2    | FileCheck %s -check-prefix=AVX2
; RUN: llc < %s -march=x86-64 -mattr=avx512f | FileCheck %s -check-prefix=AVX512

define <4 x i32> @test1(<4 x i32> %a) nounwind {
; SSE2-LABEL: test1:
; SSE2: movdqa
; SSE2: psrad $31
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret

; SSSE3-LABEL: test1:
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2-LABEL: test1:
; AVX2: vpabsd
; AVX2-NEXT: ret

; AVX512-LABEL: test1:
; AVX512: vpabsd
; AVX512-NEXT: ret
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sgt <4 x i32> %a, <i32 -1, i32 -1, i32 -1, i32 -1>
        %abs = select <4 x i1> %b, <4 x i32> %a, <4 x i32> %tmp1neg
        ret <4 x i32> %abs
}

define <4 x i32> @test2(<4 x i32> %a) nounwind {
; SSE2-LABEL: test2:
; SSE2: movdqa
; SSE2: psrad $31
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret

; SSSE3-LABEL: test2:
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2-LABEL: test2:
; AVX2: vpabsd
; AVX2-NEXT: ret

; AVX512-LABEL: test2:
; AVX512: vpabsd
; AVX512-NEXT: ret
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sge <4 x i32> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i32> %a, <4 x i32> %tmp1neg
        ret <4 x i32> %abs
}

define <8 x i16> @test3(<8 x i16> %a) nounwind {
; SSE2-LABEL: test3:
; SSE2: movdqa
; SSE2: psraw $15
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret

; SSSE3-LABEL: test3:
; SSSE3: pabsw
; SSSE3-NEXT: ret

; AVX2-LABEL: test3:
; AVX2: vpabsw
; AVX2-NEXT: ret

; AVX512-LABEL: test3:
; AVX512: vpabsw
; AVX512-NEXT: ret
        %tmp1neg = sub <8 x i16> zeroinitializer, %a
        %b = icmp sgt <8 x i16> %a, zeroinitializer
        %abs = select <8 x i1> %b, <8 x i16> %a, <8 x i16> %tmp1neg
        ret <8 x i16> %abs
}

define <16 x i8> @test4(<16 x i8> %a) nounwind {
; SSE2-LABEL: test4:
; SSE2: pxor
; SSE2: pcmpgtb
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret

; SSSE3-LABEL: test4:
; SSSE3: pabsb
; SSSE3-NEXT: ret

; AVX2-LABEL: test4:
; AVX2: vpabsb
; AVX2-NEXT: ret

; AVX512-LABEL: test4:
; AVX512: vpabsb
; AVX512-NEXT: ret
        %tmp1neg = sub <16 x i8> zeroinitializer, %a
        %b = icmp slt <16 x i8> %a, zeroinitializer
        %abs = select <16 x i1> %b, <16 x i8> %tmp1neg, <16 x i8> %a
        ret <16 x i8> %abs
}

define <4 x i32> @test5(<4 x i32> %a) nounwind {
; SSE2-LABEL: test5:
; SSE2: movdqa
; SSE2: psrad $31
; SSE2-NEXT: padd
; SSE2-NEXT: pxor
; SSE2-NEXT: ret

; SSSE3-LABEL: test5:
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2-LABEL: test5:
; AVX2: vpabsd
; AVX2-NEXT: ret

; AVX512-LABEL: test5:
; AVX512: vpabsd
; AVX512-NEXT: ret
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sle <4 x i32> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i32> %tmp1neg, <4 x i32> %a
        ret <4 x i32> %abs
}

define <8 x i32> @test6(<8 x i32> %a) nounwind {
; SSSE3-LABEL: test6:
; SSSE3: pabsd
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2-LABEL: test6:
; AVX2: vpabsd {{.*}}%ymm
; AVX2-NEXT: ret

; AVX512-LABEL: test6:
; AVX512: vpabsd {{.*}}%ymm
; AVX512-NEXT: ret
        %tmp1neg = sub <8 x i32> zeroinitializer, %a
        %b = icmp sgt <8 x i32> %a, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
        %abs = select <8 x i1> %b, <8 x i32> %a, <8 x i32> %tmp1neg
        ret <8 x i32> %abs
}

define <8 x i32> @test7(<8 x i32> %a) nounwind {
; SSSE3-LABEL: test7:
; SSSE3: pabsd
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2-LABEL: test7:
; AVX2: vpabsd {{.*}}%ymm
; AVX2-NEXT: ret

; AVX512-LABEL: test7:
; AVX512: vpabsd {{.*}}%ymm
; AVX512-NEXT: ret
        %tmp1neg = sub <8 x i32> zeroinitializer, %a
        %b = icmp sge <8 x i32> %a, zeroinitializer
        %abs = select <8 x i1> %b, <8 x i32> %a, <8 x i32> %tmp1neg
        ret <8 x i32> %abs
}

define <16 x i16> @test8(<16 x i16> %a) nounwind {
; SSSE3-LABEL: test8:
; SSSE3: pabsw
; SSSE3: pabsw
; SSSE3-NEXT: ret

; AVX2-LABEL: test8:
; AVX2: vpabsw {{.*}}%ymm
; AVX2-NEXT: ret

; AVX512-LABEL: test8:
; AVX512: vpabsw {{.*}}%ymm
; AVX512-NEXT: ret
        %tmp1neg = sub <16 x i16> zeroinitializer, %a
        %b = icmp sgt <16 x i16> %a, zeroinitializer
        %abs = select <16 x i1> %b, <16 x i16> %a, <16 x i16> %tmp1neg
        ret <16 x i16> %abs
}

define <32 x i8> @test9(<32 x i8> %a) nounwind {
; SSSE3-LABEL: test9:
; SSSE3: pabsb
; SSSE3: pabsb
; SSSE3-NEXT: ret

; AVX2-LABEL: test9:
; AVX2: vpabsb {{.*}}%ymm
; AVX2-NEXT: ret

; AVX512-LABEL: test9:
; AVX512: vpabsb {{.*}}%ymm
; AVX512-NEXT: ret
        %tmp1neg = sub <32 x i8> zeroinitializer, %a
        %b = icmp slt <32 x i8> %a, zeroinitializer
        %abs = select <32 x i1> %b, <32 x i8> %tmp1neg, <32 x i8> %a
        ret <32 x i8> %abs
}

define <8 x i32> @test10(<8 x i32> %a) nounwind {
; SSSE3-LABEL: test10:
; SSSE3: pabsd
; SSSE3: pabsd
; SSSE3-NEXT: ret

; AVX2-LABEL: test10:
; AVX2: vpabsd {{.*}}%ymm
; AVX2-NEXT: ret

; AVX512-LABEL: test10:
; AVX512: vpabsd {{.*}}%ymm
; AVX512-NEXT: ret
        %tmp1neg = sub <8 x i32> zeroinitializer, %a
        %b = icmp sle <8 x i32> %a, zeroinitializer
        %abs = select <8 x i1> %b, <8 x i32> %tmp1neg, <8 x i32> %a
        ret <8 x i32> %abs
}

define <16 x i32> @test11(<16 x i32> %a) nounwind {
; AVX2-LABEL: test11:
; AVX2: vpabsd
; AVX2: vpabsd
; AVX2-NEXT: ret

; AVX512-LABEL: test11:
; AVX512: vpabsd {{.*}}%zmm
; AVX512-NEXT: ret
        %tmp1neg = sub <16 x i32> zeroinitializer, %a
        %b = icmp sle <16 x i32> %a, zeroinitializer
        %abs = select <16 x i1> %b, <16 x i32> %tmp1neg, <16 x i32> %a
        ret <16 x i32> %abs
}

define <8 x i64> @test12(<8 x i64> %a) nounwind {
; AVX2-LABEL: test12:
; AVX2: vpxor
; AVX2: vpxor
; AVX2-NEXT: ret

; AVX512-LABEL: test12:
; AVX512: vpabsq {{.*}}%zmm
; AVX512-NEXT: ret
        %tmp1neg = sub <8 x i64> zeroinitializer, %a
        %b = icmp sle <8 x i64> %a, zeroinitializer
        %abs = select <8 x i1> %b, <8 x i64> %tmp1neg, <8 x i64> %a
        ret <8 x i64> %abs
}

define <8 x i64> @test13(<8 x i64>* %a.ptr) nounwind {
; AVX2-LABEL: test13:
; AVX2: vpxor
; AVX2: vpxor
; AVX2-NEXT: ret

; AVX512-LABEL: test13:
; AVX512: vpabsq (%
; AVX512-NEXT: ret
        %a = load <8 x i64>, <8 x i64>* %a.ptr, align 8
        %tmp1neg = sub <8 x i64> zeroinitializer, %a
        %b = icmp sle <8 x i64> %a, zeroinitializer
        %abs = select <8 x i1> %b, <8 x i64> %tmp1neg, <8 x i64> %a
        ret <8 x i64> %abs
}
