; RUN: opt < %s -basicaa -licm -S | FileCheck %s
; rdar://7282591

%struct.CFRuntimeBase = type { i32, [4 x i8] }
%struct.XXXAffineTransform = type { float, float, float, float, float, float }
%struct.XXXContext = type { %struct.CFRuntimeBase, i32, i32, i32, i8*, %struct.XXXContextDelegate*, void (%struct.XXXContext*)*, void (%struct.XXXContext*)*, %struct.XXXImage* (%struct.XXXContext*, %struct.XXXRect*, %struct.XXXImage*, i8*)*, i8*, %struct.__CFDictionary*, i32, %struct.XXXGState*, %struct.XXXGStack*, %struct.XXXRenderingState*, %struct.XXXAffineTransform, %struct.XXXPath*, %struct.__CFDictionary*, %struct.XXXPixelAccess* }
%struct.XXXContextDelegate = type opaque
%struct.XXXGStack = type opaque
%struct.XXXGState = type opaque
%struct.XXXImage = type opaque
%struct.XXXPath = type opaque
%struct.XXXPixelAccess = type opaque
%struct.XXXPoint = type { float, float }
%struct.XXXRect = type { %struct.XXXPoint, %struct.XXXPoint }
%struct.XXXRenderingState = type opaque
%struct.__CFDictionary = type opaque

define void @t(%struct.XXXContext* %context, i16* %glyphs, %struct.XXXPoint* %advances, i32 %count) nounwind optsize ssp {
; CHECK: @t
; CHECK: bb21.preheader:
; CHECK: %tmp28 = getelementptr
; CHECK: %tmp28.promoted = load
entry:
  br i1 undef, label %bb1, label %bb

bb:                                               ; preds = %entry
  br i1 undef, label %bb2, label %bb1

bb1:                                              ; preds = %bb, %entry
  ret void

bb2:                                              ; preds = %bb
  br i1 undef, label %bb35, label %bb7

bb7:                                              ; preds = %bb2
  br i1 undef, label %bb35, label %bb10

bb10:                                             ; preds = %bb7
  %tmp18 = alloca i8, i32 undef, align 1          ; <i8*> [#uses=1]
  br i1 undef, label %bb35, label %bb15

bb15:                                             ; preds = %bb10
  br i1 undef, label %bb17, label %bb16

bb16:                                             ; preds = %bb15
  %tmp21 = bitcast i8* %tmp18 to %struct.XXXPoint* ; <%struct.XXXPoint*> [#uses=1]
  br label %bb18

bb17:                                             ; preds = %bb15
  %tmp22 = malloc %struct.XXXPoint, i32 %count     ; <%struct.XXXPoint*> [#uses=1]
  br label %bb18

bb18:                                             ; preds = %bb17, %bb16
  %positions.0 = phi %struct.XXXPoint* [ %tmp21, %bb16 ], [ %tmp22, %bb17 ] ; <%struct.XXXPoint*> [#uses=1]
  br i1 undef, label %bb35, label %bb20

bb20:                                             ; preds = %bb18
  br i1 undef, label %bb21, label %bb25

bb21:                                             ; preds = %bb21, %bb20
  %tmp28 = getelementptr inbounds %struct.XXXPoint* %positions.0, i32 undef, i32 0 ; <float*> [#uses=1]
  store float undef, float* %tmp28, align 4
  %elt22 = getelementptr inbounds %struct.XXXPoint* %advances, i32 undef, i32 1 ; <float*> [#uses=1]
  %val23 = load float* %elt22                     ; <float> [#uses=0]
  br i1 undef, label %bb21, label %bb25

bb25:                                             ; preds = %bb21, %bb20
  switch i32 undef, label %bb26 [
    i32 4, label %bb27
    i32 5, label %bb27
    i32 6, label %bb27
    i32 7, label %bb28
  ]

bb26:                                             ; preds = %bb25
  unreachable

bb27:                                             ; preds = %bb25, %bb25, %bb25
  unreachable

bb28:                                             ; preds = %bb25
  unreachable

bb35:                                             ; preds = %bb18, %bb10, %bb7, %bb2
  ret void
}
