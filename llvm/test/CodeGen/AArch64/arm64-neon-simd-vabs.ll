; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK: testv4i32
; CHECK: sabd	v0.4s, v0.4s, v1.4s
define void @testv4i32(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* noalias nocapture readonly %c){
  %1 = bitcast i32* %b to <4 x i32>*
  %2 = load <4 x i32>* %1, align 4
  %3 = bitcast i32* %c to <4 x i32>*
  %4 = load <4 x i32>* %3, align 4
  %5 = sub nsw <4 x i32> %2, %4
  %6 = icmp sgt <4 x i32> %5, <i32 -1, i32 -1, i32 -1, i32 -1>
  %7 = sub <4 x i32> zeroinitializer, %5
  %8 = select <4 x i1> %6, <4 x i32> %5, <4 x i32> %7
  %9 = bitcast i32* %a to <4 x i32>*
  store <4 x i32> %8, <4 x i32>* %9, align 4
  ret void
}

; CHECK: testv2i32
; CHECK: sabd	v0.2s, v0.2s, v1.2s
define void @testv2i32(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* noalias nocapture readonly %c){
  %1 = bitcast i32* %b to <2 x i32>*
  %2 = load <2 x i32>* %1, align 4
  %3 = bitcast i32* %c to <2 x i32>*
  %4 = load <2 x i32>* %3, align 4
  %5 = sub nsw <2 x i32> %2, %4
  %6 = icmp sgt <2 x i32> %5, <i32 -1, i32 -1>
  %7 = sub <2 x i32> zeroinitializer, %5
  %8 = select <2 x i1> %6, <2 x i32> %5, <2 x i32> %7
  %9 = bitcast i32* %a to <2 x i32>*
  store <2 x i32> %8, <2 x i32>* %9, align 4
  ret void
}

; CHECK: testv8i16
; CHECK: sabd	v0.8h, v0.8h, v1.8h
define void @testv8i16(i16* noalias nocapture %a, i16* noalias nocapture readonly %b, i16* noalias nocapture readonly %c){
  %1 = bitcast i16* %b to <8 x i16>*
  %2 = load <8 x i16>* %1, align 4
  %3 = bitcast i16* %c to <8 x i16>*
  %4 = load <8 x i16>* %3, align 4
  %5 = sub nsw <8 x i16> %2, %4
  %6 = icmp sgt <8 x i16> %5,  <i16 -1, i16 -1,i16 -1, i16 -1,i16 -1, i16 -1,i16 -1, i16 -1>
  %7 = sub <8 x i16> zeroinitializer, %5
  %8 = select <8 x i1> %6, <8 x i16> %5, <8 x i16> %7
  %9 = bitcast i16* %a to <8 x i16>*
  store <8 x i16> %8, <8 x i16>* %9, align 4
  ret void
}

; CHECK: testv4i16
; CHECK: sabd	v0.4h, v0.4h, v1.4h
define void @testv4i16(i16* noalias nocapture %a, i16* noalias nocapture readonly %b, i16* noalias nocapture readonly %c){
  %1 = bitcast i16* %b to <4 x i16>*
  %2 = load <4 x i16>* %1, align 4
  %3 = bitcast i16* %c to <4 x i16>*
  %4 = load <4 x i16>* %3, align 4
  %5 = sub nsw <4 x i16> %2, %4
  %6 = icmp sgt <4 x i16> %5,  <i16 -1, i16 -1,i16 -1, i16 -1>
  %7 = sub <4 x i16> zeroinitializer, %5
  %8 = select <4 x i1> %6, <4 x i16> %5, <4 x i16> %7
  %9 = bitcast i16* %a to <4 x i16>*
  store <4 x i16> %8, <4 x i16>* %9, align 4
  ret void
}


; CHECK: testv16i8
; CHECK: sabd	v0.16b, v0.16b, v1.16b
define void @testv16i8(i8* noalias nocapture %a, i8* noalias nocapture readonly %b, i8* noalias nocapture readonly %c){
  %1 = bitcast i8* %b to <16 x i8>*
  %2 = load <16 x i8>* %1, align 4
  %3 = bitcast i8* %c to <16 x i8>*
  %4 = load <16 x i8>* %3, align 4
  %5 = sub nsw <16 x i8> %2, %4
  %6 = icmp sgt <16 x i8> %5,  <i8 -1, i8 -1,i8 -1, i8 -1,i8 -1, i8 -1,i8 -1, i8 -1,i8 -1, i8 -1,i8 -1, i8 -1,i8 -1, i8 -1,i8 -1, i8 -1>
  %7 = sub <16 x i8> zeroinitializer, %5
  %8 = select <16 x i1> %6, <16 x i8> %5, <16 x i8> %7
  %9 = bitcast i8* %a to <16 x i8>*
  store <16 x i8> %8, <16 x i8>* %9, align 4
  ret void
}

; CHECK: testv8i8
; CHECK: sabd	v0.8b, v0.8b, v1.8b
define void @testv8i8(i8* noalias nocapture %a, i8* noalias nocapture readonly %b, i8* noalias nocapture readonly %c){
  %1 = bitcast i8* %b to <8 x i8>*
  %2 = load <8 x i8>* %1, align 4
  %3 = bitcast i8* %c to <8 x i8>*
  %4 = load <8 x i8>* %3, align 4
  %5 = sub nsw <8 x i8> %2, %4
  %6 = icmp sgt <8 x i8> %5,  <i8 -1, i8 -1,i8 -1, i8 -1,i8 -1, i8 -1,i8 -1, i8 -1>
  %7 = sub <8 x i8> zeroinitializer, %5
  %8 = select <8 x i1> %6, <8 x i8> %5, <8 x i8> %7
  %9 = bitcast i8* %a to <8 x i8>*
  store <8 x i8> %8, <8 x i8>* %9, align 4
  ret void
}

