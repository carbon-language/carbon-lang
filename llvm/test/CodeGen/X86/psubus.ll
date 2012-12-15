; RUN: llc -mcpu=core2 < %s | FileCheck %s -check-prefix=SSE2
; RUN: llc -mcpu=corei7-avx < %s | FileCheck %s -check-prefix=AVX1
; RUN: llc -mcpu=core-avx2 < %s | FileCheck %s -check-prefix=AVX2

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define void @test1(i16* nocapture %head) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i16* %head, i64 %index
  %1 = bitcast i16* %0 to <8 x i16>*
  %2 = load <8 x i16>* %1, align 2
  %3 = icmp slt <8 x i16> %2, zeroinitializer
  %4 = xor <8 x i16> %2, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %5 = select <8 x i1> %3, <8 x i16> %4, <8 x i16> zeroinitializer
  store <8 x i16> %5, <8 x i16>* %1, align 2
  %index.next = add i64 %index, 8
  %6 = icmp eq i64 %index.next, 16384
  br i1 %6, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: @test1
; SSE2: psubusw LCPI0_0(%rip), %xmm0

; AVX1: @test1
; AVX1: vpsubusw LCPI0_0(%rip), %xmm0, %xmm0

; AVX2: @test1
; AVX2: vpsubusw LCPI0_0(%rip), %xmm0, %xmm0
}

define void @test2(i16* nocapture %head) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i16* %head, i64 %index
  %1 = bitcast i16* %0 to <8 x i16>*
  %2 = load <8 x i16>* %1, align 2
  %3 = icmp ugt <8 x i16> %2, <i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766>
  %4 = add <8 x i16> %2, <i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767>
  %5 = select <8 x i1> %3, <8 x i16> %4, <8 x i16> zeroinitializer
  store <8 x i16> %5, <8 x i16>* %1, align 2
  %index.next = add i64 %index, 8
  %6 = icmp eq i64 %index.next, 16384
  br i1 %6, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: @test2
; SSE2: psubusw LCPI1_0(%rip), %xmm0

; AVX1: @test2
; AVX1: vpsubusw LCPI1_0(%rip), %xmm0, %xmm0

; AVX2: @test2
; AVX2: vpsubusw LCPI1_0(%rip), %xmm0, %xmm0
}

define void @test3(i16* nocapture %head, i16 zeroext %w) nounwind {
vector.ph:
  %0 = insertelement <8 x i16> undef, i16 %w, i32 0
  %broadcast15 = shufflevector <8 x i16> %0, <8 x i16> undef, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds i16* %head, i64 %index
  %2 = bitcast i16* %1 to <8 x i16>*
  %3 = load <8 x i16>* %2, align 2
  %4 = icmp ult <8 x i16> %3, %broadcast15
  %5 = sub <8 x i16> %3, %broadcast15
  %6 = select <8 x i1> %4, <8 x i16> zeroinitializer, <8 x i16> %5
  store <8 x i16> %6, <8 x i16>* %2, align 2
  %index.next = add i64 %index, 8
  %7 = icmp eq i64 %index.next, 16384
  br i1 %7, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: @test3
; SSE2: psubusw %xmm0, %xmm1

; AVX1: @test3
; AVX1: vpsubusw %xmm0, %xmm1, %xmm1

; AVX2: @test3
; AVX2: vpsubusw %xmm0, %xmm1, %xmm1
}

define void @test4(i8* nocapture %head) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i8* %head, i64 %index
  %1 = bitcast i8* %0 to <16 x i8>*
  %2 = load <16 x i8>* %1, align 1
  %3 = icmp slt <16 x i8> %2, zeroinitializer
  %4 = xor <16 x i8> %2, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  %5 = select <16 x i1> %3, <16 x i8> %4, <16 x i8> zeroinitializer
  store <16 x i8> %5, <16 x i8>* %1, align 1
  %index.next = add i64 %index, 16
  %6 = icmp eq i64 %index.next, 16384
  br i1 %6, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: @test4
; SSE2: psubusb LCPI3_0(%rip), %xmm0

; AVX1: @test4
; AVX1: vpsubusb LCPI3_0(%rip), %xmm0, %xmm0

; AVX2: @test4
; AVX2: vpsubusb LCPI3_0(%rip), %xmm0, %xmm0
}

define void @test5(i8* nocapture %head) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i8* %head, i64 %index
  %1 = bitcast i8* %0 to <16 x i8>*
  %2 = load <16 x i8>* %1, align 1
  %3 = icmp ugt <16 x i8> %2, <i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126>
  %4 = add <16 x i8> %2, <i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127>
  %5 = select <16 x i1> %3, <16 x i8> %4, <16 x i8> zeroinitializer
  store <16 x i8> %5, <16 x i8>* %1, align 1
  %index.next = add i64 %index, 16
  %6 = icmp eq i64 %index.next, 16384
  br i1 %6, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: @test5
; SSE2: psubusb LCPI4_0(%rip), %xmm0

; AVX1: @test5
; AVX1: vpsubusb LCPI4_0(%rip), %xmm0, %xmm0

; AVX2: @test5
; AVX2: vpsubusb LCPI4_0(%rip), %xmm0, %xmm0
}

define void @test6(i8* nocapture %head, i8 zeroext %w) nounwind {
vector.ph:
  %0 = insertelement <16 x i8> undef, i8 %w, i32 0
  %broadcast15 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds i8* %head, i64 %index
  %2 = bitcast i8* %1 to <16 x i8>*
  %3 = load <16 x i8>* %2, align 1
  %4 = icmp ult <16 x i8> %3, %broadcast15
  %5 = sub <16 x i8> %3, %broadcast15
  %6 = select <16 x i1> %4, <16 x i8> zeroinitializer, <16 x i8> %5
  store <16 x i8> %6, <16 x i8>* %2, align 1
  %index.next = add i64 %index, 16
  %7 = icmp eq i64 %index.next, 16384
  br i1 %7, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: @test6
; SSE2: psubusb %xmm0, %xmm1

; AVX1: @test6
; AVX1: vpsubusb %xmm0, %xmm1, %xmm1

; AVX2: @test6
; AVX2: vpsubusb %xmm0, %xmm1, %xmm1
}

define void @test7(i16* nocapture %head) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i16* %head, i64 %index
  %1 = bitcast i16* %0 to <16 x i16>*
  %2 = load <16 x i16>* %1, align 2
  %3 = icmp slt <16 x i16> %2, zeroinitializer
  %4 = xor <16 x i16> %2, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %5 = select <16 x i1> %3, <16 x i16> %4, <16 x i16> zeroinitializer
  store <16 x i16> %5, <16 x i16>* %1, align 2
  %index.next = add i64 %index, 8
  %6 = icmp eq i64 %index.next, 16384
  br i1 %6, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: @test7
; AVX2: vpsubusw LCPI6_0(%rip), %ymm0, %ymm0
}

define void @test8(i16* nocapture %head) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i16* %head, i64 %index
  %1 = bitcast i16* %0 to <16 x i16>*
  %2 = load <16 x i16>* %1, align 2
  %3 = icmp ugt <16 x i16> %2, <i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766>
  %4 = add <16 x i16> %2, <i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767>
  %5 = select <16 x i1> %3, <16 x i16> %4, <16 x i16> zeroinitializer
  store <16 x i16> %5, <16 x i16>* %1, align 2
  %index.next = add i64 %index, 8
  %6 = icmp eq i64 %index.next, 16384
  br i1 %6, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: @test8
; AVX2: vpsubusw LCPI7_0(%rip), %ymm0, %ymm0
}

define void @test9(i16* nocapture %head, i16 zeroext %w) nounwind {
vector.ph:
  %0 = insertelement <16 x i16> undef, i16 %w, i32 0
  %broadcast15 = shufflevector <16 x i16> %0, <16 x i16> undef, <16 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds i16* %head, i64 %index
  %2 = bitcast i16* %1 to <16 x i16>*
  %3 = load <16 x i16>* %2, align 2
  %4 = icmp ult <16 x i16> %3, %broadcast15
  %5 = sub <16 x i16> %3, %broadcast15
  %6 = select <16 x i1> %4, <16 x i16> zeroinitializer, <16 x i16> %5
  store <16 x i16> %6, <16 x i16>* %2, align 2
  %index.next = add i64 %index, 8
  %7 = icmp eq i64 %index.next, 16384
  br i1 %7, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void


; AVX2: @test9
; AVX2: vpsubusw %ymm0, %ymm1, %ymm1
}

define void @test10(i8* nocapture %head) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i8* %head, i64 %index
  %1 = bitcast i8* %0 to <32 x i8>*
  %2 = load <32 x i8>* %1, align 1
  %3 = icmp slt <32 x i8> %2, zeroinitializer
  %4 = xor <32 x i8> %2, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  %5 = select <32 x i1> %3, <32 x i8> %4, <32 x i8> zeroinitializer
  store <32 x i8> %5, <32 x i8>* %1, align 1
  %index.next = add i64 %index, 16
  %6 = icmp eq i64 %index.next, 16384
  br i1 %6, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void


; AVX2: @test10
; AVX2: vpsubusb LCPI9_0(%rip), %ymm0, %ymm0
}

define void @test11(i8* nocapture %head) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i8* %head, i64 %index
  %1 = bitcast i8* %0 to <32 x i8>*
  %2 = load <32 x i8>* %1, align 1
  %3 = icmp ugt <32 x i8> %2, <i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126>
  %4 = add <32 x i8> %2, <i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127>
  %5 = select <32 x i1> %3, <32 x i8> %4, <32 x i8> zeroinitializer
  store <32 x i8> %5, <32 x i8>* %1, align 1
  %index.next = add i64 %index, 16
  %6 = icmp eq i64 %index.next, 16384
  br i1 %6, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: @test11
; AVX2: vpsubusb LCPI10_0(%rip), %ymm0, %ymm0
}

define void @test12(i8* nocapture %head, i8 zeroext %w) nounwind {
vector.ph:
  %0 = insertelement <32 x i8> undef, i8 %w, i32 0
  %broadcast15 = shufflevector <32 x i8> %0, <32 x i8> undef, <32 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds i8* %head, i64 %index
  %2 = bitcast i8* %1 to <32 x i8>*
  %3 = load <32 x i8>* %2, align 1
  %4 = icmp ult <32 x i8> %3, %broadcast15
  %5 = sub <32 x i8> %3, %broadcast15
  %6 = select <32 x i1> %4, <32 x i8> zeroinitializer, <32 x i8> %5
  store <32 x i8> %6, <32 x i8>* %2, align 1
  %index.next = add i64 %index, 16
  %7 = icmp eq i64 %index.next, 16384
  br i1 %7, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: @test12
; AVX2: vpsubusb %ymm0, %ymm1, %ymm1
}
