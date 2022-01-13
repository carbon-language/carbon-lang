; RUN: opt < %s -analyze -enable-new-pm=0 -basic-aa -da
; RUN: opt < %s -passes="print<da>"

; Test that the dependence analysis pass does seg-fault due to a null pointer
; dereference. The code in gcdMIVTest requires a null check for the result of
; getConstantPart.

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"

%0 = type { i32 }
%1 = type { [2 x [512 x %0]], [512 x %0], %2, [144 x i8], %9, %10, %11, %12, %17, [12 x i8], %18, %19, %21, [128 x i8] }
%2 = type { [64 x i16], [64 x i16], [64 x %0], [64 x %0], [128 x %0], [128 x %0], [256 x %0], [256 x %0], [32 x %0], [32 x %0], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], %3, %4, %5, [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], %6, %7, [32 x i32], [32 x i32], [32 x i32], [64 x i16], %8, [8 x i64], [4 x i64], [2 x i64], [256 x i8], [256 x i32], [64 x i16], [64 x i16] }
%3 = type { [64 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [64 x i32], [64 x i32], [64 x i32], [64 x i32], [64 x i32], [64 x i32], [64 x i32], [64 x i32], [64 x i32], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [32 x %0], [32 x %0], [128 x i8] }
%4 = type { [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16] }
%5 = type { [128 x i8], [256 x i8], [256 x i8] }
%6 = type { [64 x i32], [128 x i16], [64 x i16], [64 x i16], [64 x i16] }
%7 = type { [192 x %0], [192 x %0], [384 x %0], [1984 x %0] }
%8 = type { [128 x i8], [128 x i8], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16], [64 x i16] }
%9 = type { [32 x %0], [32 x %0], [64 x i32], [64 x i32], [64 x i32], [64 x i32] }
%10 = type { [1536 x %0], [2048 x %0], [512 x i32], [256 x i32], [32 x %0], [64 x i32], [128 x i8], [512 x i32], [1024 x %0] }
%11 = type { [512 x i32], [512 x i32], [1024 x %0], [512 x i32], [512 x %0] }
%12 = type { %13, [2048 x %0], [2048 x %0], [256 x i32], [1024 x i32], %14, [512 x %0], [256 x i32], %15, [4 x [256 x %0]], [4 x [256 x %0]], [256 x i32], [8 x [32 x i32]], [8 x [32 x %0]], [384 x %0], [256 x i32], %16 }
%13 = type { [2048 x %0] }
%14 = type { [1024 x %0], [1024 x %0] }
%15 = type { [256 x %0], [256 x %0] }
%16 = type { [128 x %0], [128 x %0] }
%17 = type { [32 x %0], [32 x i32], [32 x i32], [32 x i32], [32 x i32], [2 x [8 x [32 x %0]]], [512 x %0], [512 x %0], [58 x i16] }
%18 = type { [512 x i8] }
%19 = type { [2048 x %0], [2560 x i16], %20, [512 x i32], [256 x i32], [512 x i8] }
%20 = type { [768 x i32] }
%21 = type { [416 x i32] }

define void @test(%1* %A) #0 align 2 {
entry:
  %v1 = load i32, i32* undef, align 4
  br label %bb13

bb13:
  %v2 = phi i32 [ undef, %entry ], [ %v39, %bb38 ]
  br i1 undef, label %bb15, label %bb38

bb15:
  %v3 = mul nsw i32 %v2, undef
  br label %bb17

bb17:
  br i1 undef, label %bb21, label %bb37

bb21:
  %v22 = add nsw i32 undef, 1
  %v23 = add i32 %v22, %v3
  %v24 = mul nsw i32 %v23, %v1
  %v25 = getelementptr inbounds %1, %1* %A, i32 0, i32 7, i32 1, i32 %v24
  %v26 = bitcast %0* %v25 to <32 x i32>*
  %v27 = load <32 x i32>, <32 x i32>* %v26, align 256
  %v28 = add i32 undef, %v3
  %v29 = mul nsw i32 %v28, 32
  %v30 = getelementptr inbounds %1, %1* %A, i32 0, i32 7, i32 14, i32 %v29
  %v31 = bitcast %0* %v30 to <32 x i32>*
  %v32 = load <32 x i32>, <32 x i32>* %v31, align 128
  br i1 undef, label %bb21, label %bb37

bb37:
  br i1 undef, label %bb17, label %bb38

bb38:
  %v39 = add nsw i32 %v2, 1
  br label %bb13

bb40:
  ret void
}

attributes #0 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
