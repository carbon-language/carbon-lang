; RUN: llc -march=x86-64 -mcpu=core2 < %s | FileCheck %s -check-prefix=SSE2
; RUN: llc -march=x86-64 -mcpu=corei7 < %s | FileCheck %s -check-prefix=SSE4
; RUN: llc -march=x86-64 -mcpu=corei7-avx < %s | FileCheck %s -check-prefix=AVX1
; RUN: llc -march=x86-64 -mcpu=core-avx2 -mattr=+avx2 < %s | FileCheck %s -check-prefix=AVX2
; RUN: llc -march=x86-64 -mcpu=knl < %s | FileCheck %s  -check-prefix=AVX2 -check-prefix=AVX512F
; RUN: llc -march=x86-64 -mcpu=skx < %s | FileCheck %s  -check-prefix=AVX512BW -check-prefix=AVX512VL -check-prefix=AVX512F

define void @test1(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp slt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test1:
; SSE4: pminsb

; AVX1-LABEL: test1:
; AVX1: vpminsb

; AVX2-LABEL: test1:
; AVX2: vpminsb

; AVX512VL-LABEL: test1:
; AVX512VL: vpminsb
}

define void @test2(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp sle <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test2:
; SSE4: pminsb

; AVX1-LABEL: test2:
; AVX1: vpminsb

; AVX2-LABEL: test2:
; AVX2: vpminsb

; AVX512VL-LABEL: test2:
; AVX512VL: vpminsb
}

define void @test3(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp sgt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test3:
; SSE4: pmaxsb

; AVX1-LABEL: test3:
; AVX1: vpmaxsb

; AVX2-LABEL: test3:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test3:
; AVX512VL: vpmaxsb
}

define void @test4(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp sge <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test4:
; SSE4: pmaxsb

; AVX1-LABEL: test4:
; AVX1: vpmaxsb

; AVX2-LABEL: test4:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test4:
; AVX512VL: vpmaxsb
}

define void @test5(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp ult <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test5:
; SSE2: pminub

; AVX1-LABEL: test5:
; AVX1: vpminub

; AVX2-LABEL: test5:
; AVX2: vpminub

; AVX512VL-LABEL: test5:
; AVX512VL: vpminub 
}

define void @test6(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp ule <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test6:
; SSE2: pminub

; AVX1-LABEL: test6:
; AVX1: vpminub

; AVX2-LABEL: test6:
; AVX2: vpminub

; AVX512VL-LABEL: test6:
; AVX512VL: vpminub
}

define void @test7(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp ugt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test7:
; SSE2: pmaxub

; AVX1-LABEL: test7:
; AVX1: vpmaxub

; AVX2-LABEL: test7:
; AVX2: vpmaxub

; AVX512VL-LABEL: test7:
; AVX512VL: vpmaxub
}

define void @test8(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp uge <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.a, <16 x i8> %load.b
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test8:
; SSE2: pmaxub

; AVX1-LABEL: test8:
; AVX1: vpmaxub

; AVX2-LABEL: test8:
; AVX2: vpmaxub

; AVX512VL-LABEL: test8:
; AVX512VL: vpmaxub
}

define void @test9(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp slt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test9:
; SSE2: pminsw

; AVX1-LABEL: test9:
; AVX1: vpminsw

; AVX2-LABEL: test9:
; AVX2: vpminsw

; AVX512VL-LABEL: test9:
; AVX512VL: vpminsw 
}

define void @test10(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp sle <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test10:
; SSE2: pminsw

; AVX1-LABEL: test10:
; AVX1: vpminsw

; AVX2-LABEL: test10:
; AVX2: vpminsw

; AVX512VL-LABEL: test10:
; AVX512VL: vpminsw
}

define void @test11(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp sgt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test11:
; SSE2: pmaxsw

; AVX1-LABEL: test11:
; AVX1: vpmaxsw

; AVX2-LABEL: test11:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test11:
; AVX512VL: vpmaxsw
}

define void @test12(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp sge <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test12:
; SSE2: pmaxsw

; AVX1-LABEL: test12:
; AVX1: vpmaxsw

; AVX2-LABEL: test12:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test12:
; AVX512VL: vpmaxsw
}

define void @test13(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp ult <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test13:
; SSE4: pminuw

; AVX1-LABEL: test13:
; AVX1: vpminuw

; AVX2-LABEL: test13:
; AVX2: vpminuw

; AVX512VL-LABEL: test13:
; AVX512VL: vpminuw
}

define void @test14(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp ule <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test14:
; SSE4: pminuw

; AVX1-LABEL: test14:
; AVX1: vpminuw

; AVX2-LABEL: test14:
; AVX2: vpminuw

; AVX512VL-LABEL: test14:
; AVX512VL: vpminuw 
}

define void @test15(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp ugt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test15:
; SSE4: pmaxuw

; AVX1-LABEL: test15:
; AVX1: vpmaxuw

; AVX2-LABEL: test15:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test15:
; AVX512VL: vpmaxuw
}

define void @test16(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp uge <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.a, <8 x i16> %load.b
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test16:
; SSE4: pmaxuw

; AVX1-LABEL: test16:
; AVX1: vpmaxuw

; AVX2-LABEL: test16:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test16:
; AVX512VL: vpmaxuw
}

define void @test17(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp slt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test17:
; SSE4: pminsd

; AVX1-LABEL: test17:
; AVX1: vpminsd

; AVX2-LABEL: test17:
; AVX2: vpminsd

; AVX512VL-LABEL: test17:
; AVX512VL: vpminsd
}

define void @test18(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp sle <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test18:
; SSE4: pminsd

; AVX1-LABEL: test18:
; AVX1: vpminsd

; AVX2-LABEL: test18:
; AVX2: vpminsd

; AVX512VL-LABEL: test18:
; AVX512VL: vpminsd
}

define void @test19(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp sgt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test19:
; SSE4: pmaxsd

; AVX1-LABEL: test19:
; AVX1: vpmaxsd

; AVX2-LABEL: test19:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test19:
; AVX512VL: vpmaxsd
}

define void @test20(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp sge <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test20:
; SSE4: pmaxsd

; AVX1-LABEL: test20:
; AVX1: vpmaxsd

; AVX2-LABEL: test20:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test20:
; AVX512VL: vpmaxsd
}

define void @test21(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp ult <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test21:
; SSE4: pminud

; AVX1-LABEL: test21:
; AVX1: vpminud

; AVX2-LABEL: test21:
; AVX2: vpminud

; AVX512VL-LABEL: test21:
; AVX512VL: vpminud
}

define void @test22(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp ule <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test22:
; SSE4: pminud

; AVX1-LABEL: test22:
; AVX1: vpminud

; AVX2-LABEL: test22:
; AVX2: vpminud

; AVX512VL-LABEL: test22:
; AVX512VL: vpminud
}

define void @test23(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp ugt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test23:
; SSE4: pmaxud

; AVX1-LABEL: test23:
; AVX1: vpmaxud

; AVX2-LABEL: test23:
; AVX2: vpmaxud

; AVX512VL-LABEL: test23:
; AVX512VL: vpmaxud
}

define void @test24(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp uge <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.a, <4 x i32> %load.b
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test24:
; SSE4: pmaxud

; AVX1-LABEL: test24:
; AVX1: vpmaxud

; AVX2-LABEL: test24:
; AVX2: vpmaxud

; AVX512VL-LABEL: test24:
; AVX512VL: vpmaxud
}

define void @test25(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp slt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test25:
; AVX2: vpminsb

; AVX512VL-LABEL: test25:
; AVX512VL: vpminsb
}

define void @test26(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp sle <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test26:
; AVX2: vpminsb

; AVX512VL-LABEL: test26:
; AVX512VL: vpminsb
}

define void @test27(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp sgt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test27:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test27:
; AVX512VL: vpmaxsb
}

define void @test28(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp sge <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test28:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test28:
; AVX512VL: vpmaxsb
}

define void @test29(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp ult <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test29:
; AVX2: vpminub

; AVX512VL-LABEL: test29:
; AVX512VL: vpminub
}

define void @test30(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp ule <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test30:
; AVX2: vpminub

; AVX512VL-LABEL: test30:
; AVX512VL: vpminub
}

define void @test31(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp ugt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test31:
; AVX2: vpmaxub

; AVX512VL-LABEL: test31:
; AVX512VL: vpmaxub
}

define void @test32(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp uge <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.a, <32 x i8> %load.b
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test32:
; AVX2: vpmaxub

; AVX512VL-LABEL: test32:
; AVX512VL: vpmaxub
}

define void @test33(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp slt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test33:
; AVX2: vpminsw

; AVX512VL-LABEL: test33:
; AVX512VL: vpminsw 
}

define void @test34(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp sle <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test34:
; AVX2: vpminsw

; AVX512VL-LABEL: test34:
; AVX512VL: vpminsw
}

define void @test35(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp sgt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test35:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test35:
; AVX512VL: vpmaxsw
}

define void @test36(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp sge <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test36:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test36:
; AVX512VL: vpmaxsw
}

define void @test37(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp ult <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test37:
; AVX2: vpminuw

; AVX512VL-LABEL: test37:
; AVX512VL: vpminuw
}

define void @test38(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp ule <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test38:
; AVX2: vpminuw

; AVX512VL-LABEL: test38:
; AVX512VL: vpminuw
}

define void @test39(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp ugt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test39:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test39:
; AVX512VL: vpmaxuw
}

define void @test40(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp uge <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.a, <16 x i16> %load.b
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test40:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test40:
; AVX512VL: vpmaxuw
}

define void @test41(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp slt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test41:
; AVX2: vpminsd

; AVX512VL-LABEL: test41:
; AVX512VL: vpminsd
}

define void @test42(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp sle <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test42:
; AVX2: vpminsd

; AVX512VL-LABEL: test42:
; AVX512VL: vpminsd
}

define void @test43(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp sgt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test43:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test43:
; AVX512VL: vpmaxsd
}

define void @test44(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp sge <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test44:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test44:
; AVX512VL: vpmaxsd
}

define void @test45(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp ult <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test45:
; AVX2: vpminud

; AVX512VL-LABEL: test45:
; AVX512VL: vpminud
}

define void @test46(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp ule <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test46:
; AVX2: vpminud

; AVX512VL-LABEL: test46:
; AVX512VL: vpminud
}

define void @test47(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp ugt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test47:
; AVX2: vpmaxud

; AVX512VL-LABEL: test47:
; AVX512VL: vpmaxud
}

define void @test48(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp uge <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.a, <8 x i32> %load.b
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test48:
; AVX2: vpmaxud

; AVX512VL-LABEL: test48:
; AVX512VL: vpmaxud
}

define void @test49(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp slt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test49:
; SSE4: pmaxsb

; AVX1-LABEL: test49:
; AVX1: vpmaxsb

; AVX2-LABEL: test49:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test49:
; AVX512VL: vpmaxsb
}

define void @test50(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp sle <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test50:
; SSE4: pmaxsb

; AVX1-LABEL: test50:
; AVX1: vpmaxsb

; AVX2-LABEL: test50:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test50:
; AVX512VL: vpmaxsb
}

define void @test51(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp sgt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test51:
; SSE4: pminsb

; AVX1-LABEL: test51:
; AVX1: vpminsb

; AVX2-LABEL: test51:
; AVX2: vpminsb

; AVX512VL-LABEL: test51:
; AVX512VL: vpminsb
}

define void @test52(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp sge <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test52:
; SSE4: pminsb

; AVX1-LABEL: test52:
; AVX1: vpminsb

; AVX2-LABEL: test52:
; AVX2: vpminsb

; AVX512VL-LABEL: test52:
; AVX512VL: vpminsb
}

define void @test53(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp ult <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test53:
; SSE2: pmaxub

; AVX1-LABEL: test53:
; AVX1: vpmaxub

; AVX2-LABEL: test53:
; AVX2: vpmaxub

; AVX512VL-LABEL: test53:
; AVX512VL: vpmaxub
}

define void @test54(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp ule <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test54:
; SSE2: pmaxub

; AVX1-LABEL: test54:
; AVX1: vpmaxub

; AVX2-LABEL: test54:
; AVX2: vpmaxub

; AVX512VL-LABEL: test54:
; AVX512VL: vpmaxub
}

define void @test55(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp ugt <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test55:
; SSE2: pminub

; AVX1-LABEL: test55:
; AVX1: vpminub

; AVX2-LABEL: test55:
; AVX2: vpminub

; AVX512VL-LABEL: test55:
; AVX512VL: vpminub
}

define void @test56(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <16 x i8>*
  %ptr.b = bitcast i8* %gep.b to <16 x i8>*
  %load.a = load <16 x i8>, <16 x i8>* %ptr.a, align 2
  %load.b = load <16 x i8>, <16 x i8>* %ptr.b, align 2
  %cmp = icmp uge <16 x i8> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i8> %load.b, <16 x i8> %load.a
  store <16 x i8> %sel, <16 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test56:
; SSE2: pminub

; AVX1-LABEL: test56:
; AVX1: vpminub

; AVX2-LABEL: test56:
; AVX2: vpminub

; AVX512VL-LABEL: test56:
; AVX512VL: vpminub
}

define void @test57(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp slt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test57:
; SSE2: pmaxsw

; AVX1-LABEL: test57:
; AVX1: vpmaxsw

; AVX2-LABEL: test57:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test57:
; AVX512VL: vpmaxsw
}

define void @test58(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp sle <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test58:
; SSE2: pmaxsw

; AVX1-LABEL: test58:
; AVX1: vpmaxsw

; AVX2-LABEL: test58:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test58:
; AVX512VL: vpmaxsw
}

define void @test59(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp sgt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test59:
; SSE2: pminsw

; AVX1-LABEL: test59:
; AVX1: vpminsw

; AVX2-LABEL: test59:
; AVX2: vpminsw

; AVX512VL-LABEL: test59:
; AVX512VL: vpminsw
}

define void @test60(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp sge <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE2-LABEL: test60:
; SSE2: pminsw

; AVX1-LABEL: test60:
; AVX1: vpminsw

; AVX2-LABEL: test60:
; AVX2: vpminsw

; AVX512VL-LABEL: test60:
; AVX512VL: vpminsw
}

define void @test61(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp ult <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test61:
; SSE4: pmaxuw

; AVX1-LABEL: test61:
; AVX1: vpmaxuw

; AVX2-LABEL: test61:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test61:
; AVX512VL: vpmaxuw
}

define void @test62(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp ule <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test62:
; SSE4: pmaxuw

; AVX1-LABEL: test62:
; AVX1: vpmaxuw

; AVX2-LABEL: test62:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test62:
; AVX512VL: vpmaxuw
}

define void @test63(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp ugt <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test63:
; SSE4: pminuw

; AVX1-LABEL: test63:
; AVX1: vpminuw

; AVX2-LABEL: test63:
; AVX2: vpminuw

; AVX512VL-LABEL: test63:
; AVX512VL: vpminuw
}

define void @test64(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <8 x i16>*
  %ptr.b = bitcast i16* %gep.b to <8 x i16>*
  %load.a = load <8 x i16>, <8 x i16>* %ptr.a, align 2
  %load.b = load <8 x i16>, <8 x i16>* %ptr.b, align 2
  %cmp = icmp uge <8 x i16> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i16> %load.b, <8 x i16> %load.a
  store <8 x i16> %sel, <8 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test64:
; SSE4: pminuw

; AVX1-LABEL: test64:
; AVX1: vpminuw

; AVX2-LABEL: test64:
; AVX2: vpminuw

; AVX512VL-LABEL: test64:
; AVX512VL: vpminuw
}

define void @test65(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp slt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test65:
; SSE4: pmaxsd

; AVX1-LABEL: test65:
; AVX1: vpmaxsd

; AVX2-LABEL: test65:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test65:
; AVX512VL: vpmaxsd
}

define void @test66(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp sle <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test66:
; SSE4: pmaxsd

; AVX1-LABEL: test66:
; AVX1: vpmaxsd

; AVX2-LABEL: test66:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test66:
; AVX512VL: vpmaxsd
}

define void @test67(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp sgt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test67:
; SSE4: pminsd

; AVX1-LABEL: test67:
; AVX1: vpminsd

; AVX2-LABEL: test67:
; AVX2: vpminsd

; AVX512VL-LABEL: test67:
; AVX512VL: vpminsd
}

define void @test68(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp sge <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test68:
; SSE4: pminsd

; AVX1-LABEL: test68:
; AVX1: vpminsd

; AVX2-LABEL: test68:
; AVX2: vpminsd

; AVX512VL-LABEL: test68:
; AVX512VL: vpminsd
}

define void @test69(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp ult <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test69:
; SSE4: pmaxud

; AVX1-LABEL: test69:
; AVX1: vpmaxud

; AVX2-LABEL: test69:
; AVX2: vpmaxud

; AVX512VL-LABEL: test69:
; AVX512VL: vpmaxud
}

define void @test70(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp ule <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test70:
; SSE4: pmaxud

; AVX1-LABEL: test70:
; AVX1: vpmaxud

; AVX2-LABEL: test70:
; AVX2: vpmaxud

; AVX512VL-LABEL: test70:
; AVX512VL: vpmaxud
}

define void @test71(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp ugt <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test71:
; SSE4: pminud

; AVX1-LABEL: test71:
; AVX1: vpminud

; AVX2-LABEL: test71:
; AVX2: vpminud

; AVX512VL-LABEL: test71:
; AVX512VL: vpminud
}

define void @test72(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i32>*
  %ptr.b = bitcast i32* %gep.b to <4 x i32>*
  %load.a = load <4 x i32>, <4 x i32>* %ptr.a, align 2
  %load.b = load <4 x i32>, <4 x i32>* %ptr.b, align 2
  %cmp = icmp uge <4 x i32> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i32> %load.b, <4 x i32> %load.a
  store <4 x i32> %sel, <4 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 4
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; SSE4-LABEL: test72:
; SSE4: pminud

; AVX1-LABEL: test72:
; AVX1: vpminud

; AVX2-LABEL: test72:
; AVX2: vpminud

; AVX512VL-LABEL: test72:
; AVX512VL: vpminud
}

define void @test73(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp slt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test73:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test73:
; AVX512VL: vpmaxsb
}

define void @test74(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp sle <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test74:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test74:
; AVX512VL: vpmaxsb 
}

define void @test75(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp sgt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test75:
; AVX2: vpminsb

; AVX512VL-LABEL: test75:
; AVX512VL: vpminsb
}

define void @test76(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp sge <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test76:
; AVX2: vpminsb

; AVX512VL-LABEL: test76:
; AVX512VL: vpminsb
}

define void @test77(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp ult <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test77:
; AVX2: vpmaxub

; AVX512VL-LABEL: test77:
; AVX512VL: vpmaxub
}

define void @test78(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp ule <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test78:
; AVX2: vpmaxub

; AVX512VL-LABEL: test78:
; AVX512VL: vpmaxub
}

define void @test79(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp ugt <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test79:
; AVX2: vpminub

; AVX512VL-LABEL: test79:
; AVX512VL: vpminub
}

define void @test80(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <32 x i8>*
  %ptr.b = bitcast i8* %gep.b to <32 x i8>*
  %load.a = load <32 x i8>, <32 x i8>* %ptr.a, align 2
  %load.b = load <32 x i8>, <32 x i8>* %ptr.b, align 2
  %cmp = icmp uge <32 x i8> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i8> %load.b, <32 x i8> %load.a
  store <32 x i8> %sel, <32 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test80:
; AVX2: vpminub

; AVX512VL-LABEL: test80:
; AVX512VL: vpminub
}

define void @test81(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp slt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test81:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test81:
; AVX512VL: vpmaxsw
}

define void @test82(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp sle <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test82:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test82:
; AVX512VL: vpmaxsw
}

define void @test83(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp sgt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test83:
; AVX2: vpminsw

; AVX512VL-LABEL: test83:
; AVX512VL: vpminsw
}

define void @test84(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp sge <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test84:
; AVX2: vpminsw

; AVX512VL-LABEL: test84:
; AVX512VL: vpminsw
}

define void @test85(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp ult <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test85:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test85:
; AVX512VL: vpmaxuw
}

define void @test86(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp ule <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test86:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test86:
; AVX512VL: vpmaxuw
}

define void @test87(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp ugt <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test87:
; AVX2: vpminuw

; AVX512VL-LABEL: test87:
; AVX512VL: vpminuw
}

define void @test88(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <16 x i16>*
  %ptr.b = bitcast i16* %gep.b to <16 x i16>*
  %load.a = load <16 x i16>, <16 x i16>* %ptr.a, align 2
  %load.b = load <16 x i16>, <16 x i16>* %ptr.b, align 2
  %cmp = icmp uge <16 x i16> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i16> %load.b, <16 x i16> %load.a
  store <16 x i16> %sel, <16 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test88:
; AVX2: vpminuw

; AVX512VL-LABEL: test88:
; AVX512VL: vpminuw
}

define void @test89(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp slt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test89:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test89:
; AVX512VL: vpmaxsd
}

define void @test90(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp sle <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test90:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test90:
; AVX512VL: vpmaxsd
}

define void @test91(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp sgt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test91:
; AVX2: vpminsd

; AVX512VL-LABEL: test91:
; AVX512VL: vpminsd
}

define void @test92(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp sge <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test92:
; AVX2: vpminsd

; AVX512VL-LABEL: test92:
; AVX512VL: vpminsd
}

define void @test93(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp ult <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test93:
; AVX2: vpmaxud

; AVX512VL-LABEL: test93:
; AVX512VL: vpmaxud
}

define void @test94(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp ule <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test94:
; AVX2: vpmaxud

; AVX512VL-LABEL: test94:
; AVX512VL: vpmaxud
}

define void @test95(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp ugt <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test95:
; AVX2: vpminud

; AVX512VL-LABEL: test95:
; AVX512VL: vpminud
}

define void @test96(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i32>*
  %ptr.b = bitcast i32* %gep.b to <8 x i32>*
  %load.a = load <8 x i32>, <8 x i32>* %ptr.a, align 2
  %load.b = load <8 x i32>, <8 x i32>* %ptr.b, align 2
  %cmp = icmp uge <8 x i32> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i32> %load.b, <8 x i32> %load.a
  store <8 x i32> %sel, <8 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX2-LABEL: test96:
; AVX2: vpminud

; AVX512VL-LABEL: test96:
; AVX512VL: vpminud
}

; ----------------------------

define void @test97(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp slt <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.a, <64 x i8> %load.b
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test97:
; AVX512BW: vpminsb {{.*}}
}

define void @test98(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp sle <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.a, <64 x i8> %load.b
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test98:
; AVX512BW: vpminsb {{.*}}
}

define void @test99(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp sgt <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.a, <64 x i8> %load.b
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test99:
; AVX512BW: vpmaxsb {{.*}}
}

define void @test100(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp sge <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.a, <64 x i8> %load.b
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test100:
; AVX512BW: vpmaxsb {{.*}}
}

define void @test101(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp ult <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.a, <64 x i8> %load.b
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test101:
; AVX512BW: vpminub {{.*}}
}

define void @test102(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp ule <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.a, <64 x i8> %load.b
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test102:
; AVX512BW: vpminub {{.*}}
}

define void @test103(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp ugt <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.a, <64 x i8> %load.b
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test103:
; AVX512BW: vpmaxub {{.*}}
}

define void @test104(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp uge <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.a, <64 x i8> %load.b
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test104:
; AVX512BW: vpmaxub {{.*}}
}

define void @test105(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp slt <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.a, <32 x i16> %load.b
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test105:
; AVX512BW: vpminsw {{.*}}
}

define void @test106(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp sle <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.a, <32 x i16> %load.b
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test106:
; AVX512BW: vpminsw {{.*}}
}

define void @test107(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp sgt <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.a, <32 x i16> %load.b
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test107:
; AVX512BW: vpmaxsw {{.*}}
}

define void @test108(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp sge <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.a, <32 x i16> %load.b
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test108:
; AVX512BW: vpmaxsw {{.*}}
}

define void @test109(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp ult <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.a, <32 x i16> %load.b
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test109:
; AVX512BW: vpminuw {{.*}}
}

define void @test110(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp ule <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.a, <32 x i16> %load.b
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test110:
; AVX512BW: vpminuw {{.*}}
}

define void @test111(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp ugt <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.a, <32 x i16> %load.b
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test111:
; AVX512BW: vpmaxuw {{.*}}
}

define void @test112(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp uge <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.a, <32 x i16> %load.b
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test112:
; AVX512BW: vpmaxuw {{.*}}
}

define void @test113(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp slt <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.a, <16 x i32> %load.b
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test113:
; AVX512F: vpminsd {{.*}}
}

define void @test114(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp sle <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.a, <16 x i32> %load.b
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test114:
; AVX512F: vpminsd {{.*}}
}

define void @test115(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp sgt <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.a, <16 x i32> %load.b
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test115:
; AVX512F: vpmaxsd {{.*}}
}

define void @test116(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp sge <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.a, <16 x i32> %load.b
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test116:
; AVX512F: vpmaxsd {{.*}}
}

define void @test117(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp ult <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.a, <16 x i32> %load.b
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test117:
; AVX512F: vpminud {{.*}}
}

define void @test118(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp ule <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.a, <16 x i32> %load.b
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test118:
; AVX512F: vpminud {{.*}}
}

define void @test119(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp ugt <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.a, <16 x i32> %load.b
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test119:
; AVX512F: vpmaxud {{.*}}
}

define void @test120(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp uge <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.a, <16 x i32> %load.b
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test120:
; AVX512F: vpmaxud {{.*}}
}

define void @test121(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp slt <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.a, <8 x i64> %load.b
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test121:
; AVX512F: vpminsq {{.*}}
}

define void @test122(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp sle <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.a, <8 x i64> %load.b
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test122:
; AVX512F: vpminsq {{.*}}
}

define void @test123(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp sgt <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.a, <8 x i64> %load.b
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test123:
; AVX512F: vpmaxsq {{.*}}
}

define void @test124(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp sge <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.a, <8 x i64> %load.b
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test124:
; AVX512F: vpmaxsq {{.*}}
}

define void @test125(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp ult <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.a, <8 x i64> %load.b
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test125:
; AVX512F: vpminuq {{.*}}
}

define void @test126(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp ule <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.a, <8 x i64> %load.b
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test126:
; AVX512F: vpminuq {{.*}}
}

define void @test127(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp ugt <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.a, <8 x i64> %load.b
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test127:
; AVX512F: vpmaxuq {{.*}}
}

define void @test128(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp uge <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.a, <8 x i64> %load.b
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test128:
; AVX512F: vpmaxuq {{.*}}
}

define void @test129(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp slt <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.b, <64 x i8> %load.a
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test129:
; AVX512BW: vpmaxsb
}

define void @test130(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp sle <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.b, <64 x i8> %load.a
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test130:
; AVX512BW: vpmaxsb
}

define void @test131(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp sgt <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.b, <64 x i8> %load.a
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test131:
; AVX512BW: vpminsb
}

define void @test132(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp sge <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.b, <64 x i8> %load.a
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test132:
; AVX512BW: vpminsb
}

define void @test133(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp ult <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.b, <64 x i8> %load.a
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test133:
; AVX512BW: vpmaxub
}

define void @test134(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp ule <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.b, <64 x i8> %load.a
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test134:
; AVX512BW: vpmaxub
}

define void @test135(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp ugt <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.b, <64 x i8> %load.a
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test135:
; AVX512BW: vpminub
}

define void @test136(i8* nocapture %a, i8* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i8, i8* %a, i64 %index
  %gep.b = getelementptr inbounds i8, i8* %b, i64 %index
  %ptr.a = bitcast i8* %gep.a to <64 x i8>*
  %ptr.b = bitcast i8* %gep.b to <64 x i8>*
  %load.a = load <64 x i8>, <64 x i8>* %ptr.a, align 2
  %load.b = load <64 x i8>, <64 x i8>* %ptr.b, align 2
  %cmp = icmp uge <64 x i8> %load.a, %load.b
  %sel = select <64 x i1> %cmp, <64 x i8> %load.b, <64 x i8> %load.a
  store <64 x i8> %sel, <64 x i8>* %ptr.a, align 2
  %index.next = add i64 %index, 32
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test136:
; AVX512BW: vpminub
}

define void @test137(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp slt <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.b, <32 x i16> %load.a
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test137:
; AVX512BW: vpmaxsw
}

define void @test138(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp sle <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.b, <32 x i16> %load.a
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test138:
; AVX512BW: vpmaxsw
}

define void @test139(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp sgt <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.b, <32 x i16> %load.a
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test139:
; AVX512BW: vpminsw
}

define void @test140(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp sge <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.b, <32 x i16> %load.a
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test140:
; AVX512BW: vpminsw
}

define void @test141(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp ult <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.b, <32 x i16> %load.a
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test141:
; AVX512BW: vpmaxuw
}

define void @test142(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp ule <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.b, <32 x i16> %load.a
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test142:
; AVX512BW: vpmaxuw
}

define void @test143(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp ugt <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.b, <32 x i16> %load.a
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test143:
; AVX512BW: vpminuw
}

define void @test144(i16* nocapture %a, i16* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i16, i16* %a, i64 %index
  %gep.b = getelementptr inbounds i16, i16* %b, i64 %index
  %ptr.a = bitcast i16* %gep.a to <32 x i16>*
  %ptr.b = bitcast i16* %gep.b to <32 x i16>*
  %load.a = load <32 x i16>, <32 x i16>* %ptr.a, align 2
  %load.b = load <32 x i16>, <32 x i16>* %ptr.b, align 2
  %cmp = icmp uge <32 x i16> %load.a, %load.b
  %sel = select <32 x i1> %cmp, <32 x i16> %load.b, <32 x i16> %load.a
  store <32 x i16> %sel, <32 x i16>* %ptr.a, align 2
  %index.next = add i64 %index, 16
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512BW-LABEL: test144:
; AVX512BW: vpminuw
}

define void @test145(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp slt <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.b, <16 x i32> %load.a
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test145:
; AVX512F: vpmaxsd
}

define void @test146(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp sle <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.b, <16 x i32> %load.a
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test146:
; AVX512F: vpmaxsd
}

define void @test147(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp sgt <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.b, <16 x i32> %load.a
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test147:
; AVX512F: vpminsd
}

define void @test148(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp sge <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.b, <16 x i32> %load.a
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test148:
; AVX512F: vpminsd
}

define void @test149(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp ult <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.b, <16 x i32> %load.a
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test149:
; AVX512F: vpmaxud
}

define void @test150(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp ule <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.b, <16 x i32> %load.a
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test150:
; AVX512F: vpmaxud
}

define void @test151(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp ugt <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.b, <16 x i32> %load.a
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test151:
; AVX512F: vpminud
}

define void @test152(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <16 x i32>*
  %ptr.b = bitcast i32* %gep.b to <16 x i32>*
  %load.a = load <16 x i32>, <16 x i32>* %ptr.a, align 2
  %load.b = load <16 x i32>, <16 x i32>* %ptr.b, align 2
  %cmp = icmp uge <16 x i32> %load.a, %load.b
  %sel = select <16 x i1> %cmp, <16 x i32> %load.b, <16 x i32> %load.a
  store <16 x i32> %sel, <16 x i32>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test152:
; AVX512F: vpminud
}

; -----------------------

define void @test153(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp slt <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.b, <8 x i64> %load.a
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test153:
; AVX512F: vpmaxsq
}

define void @test154(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp sle <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.b, <8 x i64> %load.a
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test154:
; AVX512F: vpmaxsq
}

define void @test155(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp sgt <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.b, <8 x i64> %load.a
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test155:
; AVX512F: vpminsq
}

define void @test156(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp sge <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.b, <8 x i64> %load.a
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test156:
; AVX512F: vpminsq
}

define void @test157(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp ult <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.b, <8 x i64> %load.a
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test157:
; AVX512F: vpmaxuq
}

define void @test158(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp ule <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.b, <8 x i64> %load.a
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test158:
; AVX512F: vpmaxuq
}

define void @test159(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp ugt <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.b, <8 x i64> %load.a
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test159:
; AVX512F: vpminuq
}

define void @test160(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <8 x i64>*
  %ptr.b = bitcast i32* %gep.b to <8 x i64>*
  %load.a = load <8 x i64>, <8 x i64>* %ptr.a, align 2
  %load.b = load <8 x i64>, <8 x i64>* %ptr.b, align 2
  %cmp = icmp uge <8 x i64> %load.a, %load.b
  %sel = select <8 x i1> %cmp, <8 x i64> %load.b, <8 x i64> %load.a
  store <8 x i64> %sel, <8 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512F-LABEL: test160:
; AVX512F: vpminuq
}

define void @test161(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp slt <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.a, <4 x i64> %load.b
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test161:
; AVX512VL: vpminsq
}

define void @test162(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp sle <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.a, <4 x i64> %load.b
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test162:
; AVX512VL: vpminsq
}

define void @test163(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp sgt <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.a, <4 x i64> %load.b
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test163:
; AVX512VL: vpmaxsq 
}

define void @test164(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp sge <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.a, <4 x i64> %load.b
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test164:
; AVX512VL: vpmaxsq
}

define void @test165(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp ult <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.a, <4 x i64> %load.b
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test165:
; AVX512VL: vpminuq 
}

define void @test166(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp ule <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.a, <4 x i64> %load.b
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test166:
; AVX512VL: vpminuq
}

define void @test167(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp ugt <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.a, <4 x i64> %load.b
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test167:
; AVX512VL: vpmaxuq
}

define void @test168(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp uge <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.a, <4 x i64> %load.b
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test168:
; AVX512VL: vpmaxuq
}

define void @test169(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp slt <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.b, <4 x i64> %load.a
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test169:
; AVX512VL: vpmaxsq
}

define void @test170(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp sle <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.b, <4 x i64> %load.a
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test170:
; AVX512VL: vpmaxsq
}

define void @test171(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp sgt <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.b, <4 x i64> %load.a
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test171:
; AVX512VL: vpminsq
}

define void @test172(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp sge <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.b, <4 x i64> %load.a
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test172:
; AVX512VL: vpminsq
}

define void @test173(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp ult <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.b, <4 x i64> %load.a
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test173:
; AVX512VL: vpmaxuq
}

define void @test174(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp ule <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.b, <4 x i64> %load.a
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test174:
; AVX512VL: vpmaxuq
}

define void @test175(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp ugt <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.b, <4 x i64> %load.a
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test175:
; AVX512VL: vpminuq
}

define void @test176(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <4 x i64>*
  %ptr.b = bitcast i32* %gep.b to <4 x i64>*
  %load.a = load <4 x i64>, <4 x i64>* %ptr.a, align 2
  %load.b = load <4 x i64>, <4 x i64>* %ptr.b, align 2
  %cmp = icmp uge <4 x i64> %load.a, %load.b
  %sel = select <4 x i1> %cmp, <4 x i64> %load.b, <4 x i64> %load.a
  store <4 x i64> %sel, <4 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test176:
; AVX512VL: vpminuq
}

define void @test177(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp slt <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.a, <2 x i64> %load.b
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test177:
; AVX512VL: vpminsq
}

define void @test178(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp sle <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.a, <2 x i64> %load.b
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test178:
; AVX512VL: vpminsq
}

define void @test179(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp sgt <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.a, <2 x i64> %load.b
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test179:
; AVX512VL: vpmaxsq
}

define void @test180(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp sge <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.a, <2 x i64> %load.b
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test180:
; AVX512VL: vpmaxsq
}

define void @test181(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp ult <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.a, <2 x i64> %load.b
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test181:
; AVX512VL: vpminuq
}

define void @test182(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp ule <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.a, <2 x i64> %load.b
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test182:
; AVX512VL: vpminuq
}

define void @test183(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp ugt <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.a, <2 x i64> %load.b
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test183:
; AVX512VL: vpmaxuq
}

define void @test184(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp uge <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.a, <2 x i64> %load.b
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test184:
; AVX512VL: vpmaxuq
}

define void @test185(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp slt <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.b, <2 x i64> %load.a
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test185:
; AVX512VL: vpmaxsq
}

define void @test186(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp sle <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.b, <2 x i64> %load.a
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test186:
; AVX512VL: vpmaxsq
}

define void @test187(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp sgt <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.b, <2 x i64> %load.a
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test187:
; AVX512VL: vpminsq
}

define void @test188(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp sge <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.b, <2 x i64> %load.a
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test188:
; AVX512VL: vpminsq
}

define void @test189(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp ult <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.b, <2 x i64> %load.a
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test189:
; AVX512VL: vpmaxuq
}

define void @test190(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp ule <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.b, <2 x i64> %load.a
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test190:
; AVX512VL: vpmaxuq
}

define void @test191(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp ugt <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.b, <2 x i64> %load.a
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test191:
; AVX512VL: vpminuq
}

define void @test192(i32* nocapture %a, i32* nocapture %b) nounwind {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %gep.a = getelementptr inbounds i32, i32* %a, i64 %index
  %gep.b = getelementptr inbounds i32, i32* %b, i64 %index
  %ptr.a = bitcast i32* %gep.a to <2 x i64>*
  %ptr.b = bitcast i32* %gep.b to <2 x i64>*
  %load.a = load <2 x i64>, <2 x i64>* %ptr.a, align 2
  %load.b = load <2 x i64>, <2 x i64>* %ptr.b, align 2
  %cmp = icmp uge <2 x i64> %load.a, %load.b
  %sel = select <2 x i1> %cmp, <2 x i64> %load.b, <2 x i64> %load.a
  store <2 x i64> %sel, <2 x i64>* %ptr.a, align 2
  %index.next = add i64 %index, 8
  %loop = icmp eq i64 %index.next, 16384
  br i1 %loop, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void

; AVX512VL-LABEL: test192:
; AVX512VL: vpminuq
}
