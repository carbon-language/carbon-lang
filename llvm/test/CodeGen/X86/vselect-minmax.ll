; RUN: llc -march=x86-64 -mcpu=core2 < %s | FileCheck %s -check-prefix=SSE2
; RUN: llc -march=x86-64 -mcpu=corei7 < %s | FileCheck %s -check-prefix=SSE4
; RUN: llc -march=x86-64 -mcpu=corei7-avx < %s | FileCheck %s -check-prefix=AVX1
; RUN: llc -march=x86-64 -mcpu=core-avx2 -mattr=+avx2 < %s | FileCheck %s -check-prefix=AVX2

define void @test1(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp slt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test1:
; SSE4: pminsb

; AVX1: test1:
; AVX1: vpminsb

; AVX2: test1:
; AVX2: vpminsb
}

define void @test2(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp sle <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test2:
; SSE4: pminsb

; AVX1: test2:
; AVX1: vpminsb

; AVX2: test2:
; AVX2: vpminsb
}

define void @test3(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp sgt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test3:
; SSE4: pmaxsb

; AVX1: test3:
; AVX1: vpmaxsb

; AVX2: test3:
; AVX2: vpmaxsb
}

define void @test4(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp sge <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test4:
; SSE4: pmaxsb

; AVX1: test4:
; AVX1: vpmaxsb

; AVX2: test4:
; AVX2: vpmaxsb
}

define void @test5(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp ult <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test5:
; SSE2: pminub

; AVX1: test5:
; AVX1: vpminub

; AVX2: test5:
; AVX2: vpminub
}

define void @test6(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp ule <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test6:
; SSE2: pminub

; AVX1: test6:
; AVX1: vpminub

; AVX2: test6:
; AVX2: vpminub
}

define void @test7(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp ugt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test7:
; SSE2: pmaxub

; AVX1: test7:
; AVX1: vpmaxub

; AVX2: test7:
; AVX2: vpmaxub
}

define void @test8(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp uge <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test8:
; SSE2: pmaxub

; AVX1: test8:
; AVX1: vpmaxub

; AVX2: test8:
; AVX2: vpmaxub
}

define void @test9(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp slt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test9:
; SSE2: pminsw

; AVX1: test9:
; AVX1: vpminsw

; AVX2: test9:
; AVX2: vpminsw
}

define void @test10(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp sle <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test10:
; SSE2: pminsw

; AVX1: test10:
; AVX1: vpminsw

; AVX2: test10:
; AVX2: vpminsw
}

define void @test11(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp sgt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test11:
; SSE2: pmaxsw

; AVX1: test11:
; AVX1: vpmaxsw

; AVX2: test11:
; AVX2: vpmaxsw
}

define void @test12(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp sge <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test12:
; SSE2: pmaxsw

; AVX1: test12:
; AVX1: vpmaxsw

; AVX2: test12:
; AVX2: vpmaxsw
}

define void @test13(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp ult <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test13:
; SSE4: pminuw

; AVX1: test13:
; AVX1: vpminuw

; AVX2: test13:
; AVX2: vpminuw
}

define void @test14(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp ule <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test14:
; SSE4: pminuw

; AVX1: test14:
; AVX1: vpminuw

; AVX2: test14:
; AVX2: vpminuw
}

define void @test15(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp ugt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test15:
; SSE4: pmaxuw

; AVX1: test15:
; AVX1: vpmaxuw

; AVX2: test15:
; AVX2: vpmaxuw
}

define void @test16(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp uge <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test16:
; SSE4: pmaxuw

; AVX1: test16:
; AVX1: vpmaxuw

; AVX2: test16:
; AVX2: vpmaxuw
}

define void @test17(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp slt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test17:
; SSE4: pminsd

; AVX1: test17:
; AVX1: vpminsd

; AVX2: test17:
; AVX2: vpminsd
}

define void @test18(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp sle <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test18:
; SSE4: pminsd

; AVX1: test18:
; AVX1: vpminsd

; AVX2: test18:
; AVX2: vpminsd
}

define void @test19(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp sgt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test19:
; SSE4: pmaxsd

; AVX1: test19:
; AVX1: vpmaxsd

; AVX2: test19:
; AVX2: vpmaxsd
}

define void @test20(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp sge <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test20:
; SSE4: pmaxsd

; AVX1: test20:
; AVX1: vpmaxsd

; AVX2: test20:
; AVX2: vpmaxsd
}

define void @test21(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp ult <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test21:
; SSE4: pminud

; AVX1: test21:
; AVX1: vpminud

; AVX2: test21:
; AVX2: vpminud
}

define void @test22(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp ule <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test22:
; SSE4: pminud

; AVX1: test22:
; AVX1: vpminud

; AVX2: test22:
; AVX2: vpminud
}

define void @test23(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp ugt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test23:
; SSE4: pmaxud

; AVX1: test23:
; AVX1: vpmaxud

; AVX2: test23:
; AVX2: vpmaxud
}

define void @test24(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp uge <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test24:
; SSE4: pmaxud

; AVX1: test24:
; AVX1: vpmaxud

; AVX2: test24:
; AVX2: vpmaxud
}

define void @test25(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp slt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test25:
; AVX2: vpminsb
}

define void @test26(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp sle <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test26:
; AVX2: vpminsb
}

define void @test27(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp sgt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test27:
; AVX2: vpmaxsb
}

define void @test28(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp sge <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test28:
; AVX2: vpmaxsb
}

define void @test29(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp ult <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test29:
; AVX2: vpminub
}

define void @test30(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp ule <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test30:
; AVX2: vpminub
}

define void @test31(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp ugt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test31:
; AVX2: vpmaxub
}

define void @test32(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp uge <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test32:
; AVX2: vpmaxub
}

define void @test33(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp slt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test33:
; AVX2: vpminsw
}

define void @test34(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp sle <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test34:
; AVX2: vpminsw
}

define void @test35(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp sgt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test35:
; AVX2: vpmaxsw
}

define void @test36(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp sge <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test36:
; AVX2: vpmaxsw
}

define void @test37(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp ult <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test37:
; AVX2: vpminuw
}

define void @test38(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp ule <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test38:
; AVX2: vpminuw
}

define void @test39(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp ugt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test39:
; AVX2: vpmaxuw
}

define void @test40(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp uge <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test40:
; AVX2: vpmaxuw
}

define void @test41(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp slt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test41:
; AVX2: vpminsd
}

define void @test42(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp sle <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test42:
; AVX2: vpminsd
}

define void @test43(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp sgt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test43:
; AVX2: vpmaxsd
}

define void @test44(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp sge <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test44:
; AVX2: vpmaxsd
}

define void @test45(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp ult <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test45:
; AVX2: vpminud
}

define void @test46(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp ule <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test46:
; AVX2: vpminud
}

define void @test47(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp ugt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test47:
; AVX2: vpmaxud
}

define void @test48(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp uge <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test48:
; AVX2: vpmaxud
}

define void @test49(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp slt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test49:
; SSE4: pmaxsb

; AVX1: test49:
; AVX1: vpmaxsb

; AVX2: test49:
; AVX2: vpmaxsb
}

define void @test50(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp sle <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test50:
; SSE4: pmaxsb

; AVX1: test50:
; AVX1: vpmaxsb

; AVX2: test50:
; AVX2: vpmaxsb
}

define void @test51(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp sgt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test51:
; SSE4: pminsb

; AVX1: test51:
; AVX1: vpminsb

; AVX2: test51:
; AVX2: vpminsb
}

define void @test52(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp sge <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test52:
; SSE4: pminsb

; AVX1: test52:
; AVX1: vpminsb

; AVX2: test52:
; AVX2: vpminsb
}

define void @test53(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp ult <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test53:
; SSE2: pmaxub

; AVX1: test53:
; AVX1: vpmaxub

; AVX2: test53:
; AVX2: vpmaxub
}

define void @test54(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp ule <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test54:
; SSE2: pmaxub

; AVX1: test54:
; AVX1: vpmaxub

; AVX2: test54:
; AVX2: vpmaxub
}

define void @test55(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp ugt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test55:
; SSE2: pminub

; AVX1: test55:
; AVX1: vpminub

; AVX2: test55:
; AVX2: vpminub
}

define void @test56(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>* %ptr.b, align 2
  %cmp = icmp uge <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test56:
; SSE2: pminub

; AVX1: test56:
; AVX1: vpminub

; AVX2: test56:
; AVX2: vpminub
}

define void @test57(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp slt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test57:
; SSE2: pmaxsw

; AVX1: test57:
; AVX1: vpmaxsw

; AVX2: test57:
; AVX2: vpmaxsw
}

define void @test58(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp sle <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test58:
; SSE2: pmaxsw

; AVX1: test58:
; AVX1: vpmaxsw

; AVX2: test58:
; AVX2: vpmaxsw
}

define void @test59(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp sgt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test59:
; SSE2: pminsw

; AVX1: test59:
; AVX1: vpminsw

; AVX2: test59:
; AVX2: vpminsw
}

define void @test60(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp sge <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2: test60:
; SSE2: pminsw

; AVX1: test60:
; AVX1: vpminsw

; AVX2: test60:
; AVX2: vpminsw
}

define void @test61(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp ult <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test61:
; SSE4: pmaxuw

; AVX1: test61:
; AVX1: vpmaxuw

; AVX2: test61:
; AVX2: vpmaxuw
}

define void @test62(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp ule <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test62:
; SSE4: pmaxuw

; AVX1: test62:
; AVX1: vpmaxuw

; AVX2: test62:
; AVX2: vpmaxuw
}

define void @test63(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp ugt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test63:
; SSE4: pminuw

; AVX1: test63:
; AVX1: vpminuw

; AVX2: test63:
; AVX2: vpminuw
}

define void @test64(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>* %ptr.b, align 2
  %cmp = icmp uge <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test64:
; SSE4: pminuw

; AVX1: test64:
; AVX1: vpminuw

; AVX2: test64:
; AVX2: vpminuw
}

define void @test65(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp slt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test65:
; SSE4: pmaxsd

; AVX1: test65:
; AVX1: vpmaxsd

; AVX2: test65:
; AVX2: vpmaxsd
}

define void @test66(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp sle <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test66:
; SSE4: pmaxsd

; AVX1: test66:
; AVX1: vpmaxsd

; AVX2: test66:
; AVX2: vpmaxsd
}

define void @test67(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp sgt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test67:
; SSE4: pminsd

; AVX1: test67:
; AVX1: vpminsd

; AVX2: test67:
; AVX2: vpminsd
}

define void @test68(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp sge <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test68:
; SSE4: pminsd

; AVX1: test68:
; AVX1: vpminsd

; AVX2: test68:
; AVX2: vpminsd
}

define void @test69(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp ult <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test69:
; SSE4: pmaxud

; AVX1: test69:
; AVX1: vpmaxud

; AVX2: test69:
; AVX2: vpmaxud
}

define void @test70(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp ule <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test70:
; SSE4: pmaxud

; AVX1: test70:
; AVX1: vpmaxud

; AVX2: test70:
; AVX2: vpmaxud
}

define void @test71(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp ugt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test71:
; SSE4: pminud

; AVX1: test71:
; AVX1: vpminud

; AVX2: test71:
; AVX2: vpminud
}

define void @test72(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>* %ptr.b, align 2
  %cmp = icmp uge <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4: test72:
; SSE4: pminud

; AVX1: test72:
; AVX1: vpminud

; AVX2: test72:
; AVX2: vpminud
}

define void @test73(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp slt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test73:
; AVX2: vpmaxsb
}

define void @test74(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp sle <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test74:
; AVX2: vpmaxsb
}

define void @test75(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp sgt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test75:
; AVX2: vpminsb
}

define void @test76(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp sge <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test76:
; AVX2: vpminsb
}

define void @test77(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp ult <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test77:
; AVX2: vpmaxub
}

define void @test78(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp ule <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test78:
; AVX2: vpmaxub
}

define void @test79(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp ugt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test79:
; AVX2: vpminub
}

define void @test80(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>* %ptr.b, align 2
  %cmp = icmp uge <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test80:
; AVX2: vpminub
}

define void @test81(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp slt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test81:
; AVX2: vpmaxsw
}

define void @test82(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp sle <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test82:
; AVX2: vpmaxsw
}

define void @test83(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp sgt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test83:
; AVX2: vpminsw
}

define void @test84(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp sge <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test84:
; AVX2: vpminsw
}

define void @test85(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp ult <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test85:
; AVX2: vpmaxuw
}

define void @test86(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp ule <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test86:
; AVX2: vpmaxuw
}

define void @test87(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp ugt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test87:
; AVX2: vpminuw
}

define void @test88(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>* %ptr.b, align 2
  %cmp = icmp uge <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test88:
; AVX2: vpminuw
}

define void @test89(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp slt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test89:
; AVX2: vpmaxsd
}

define void @test90(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp sle <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test90:
; AVX2: vpmaxsd
}

define void @test91(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp sgt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test91:
; AVX2: vpminsd
}

define void @test92(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp sge <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test92:
; AVX2: vpminsd
}

define void @test93(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp ult <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test93:
; AVX2: vpmaxud
}

define void @test94(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp ule <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test94:
; AVX2: vpmaxud
}

define void @test95(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp ugt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test95:
; AVX2: vpminud
}

define void @test96(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>* %ptr.b, align 2
  %cmp = icmp uge <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2: test96:
; AVX2: vpminud
}
