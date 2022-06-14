; RUN: opt -disable-output "-passes=print<scalar-evolution>" -S < %s 2>&1 | FileCheck %s

; The exit value from this loop was originally calculated as 0.
; The actual exit condition is 256*256 == 0 (in i16).

; CHECK: Printing analysis 'Scalar Evolution Analysis' for function 'f0':
; CHECK-NEXT: Classifying expressions for: @f0
; CHECK-NEXT:   %v1 = phi i16 [ 0, %b0 ], [ %v2, %b1 ]
; CHECK-NEXT:   -->  {0,+,-1}<%b1> U: [-255,1) S: [-255,1)            Exits: -255            LoopDispositions: { %b1: Computable }
; CHECK-NEXT:   %v2 = add i16 %v1, -1
; CHECK-NEXT:   -->  {-1,+,-1}<%b1> U: [-256,0) S: [-256,0)           Exits: -256            LoopDispositions: { %b1: Computable }
; CHECK-NEXT:   %v3 = mul i16 %v2, %v2
; CHECK-NEXT:   -->  {1,+,3,+,2}<%b1> U: full-set S: full-set         Exits: 0               LoopDispositions: { %b1: Computable }
; CHECK-NEXT:   %v5 = phi i16 [ %v2, %b1 ]
; CHECK-NEXT:   -->  {-1,+,-1}<%b1> U: [-256,0) S: [-256,0)  -->  -256 U: [-256,-255) S: [-256,-255)
; CHECK-NEXT:   %v6 = phi i16 [ %v3, %b1 ]
; CHECK-NEXT:   -->  {1,+,3,+,2}<%b1> U: full-set S: full-set  -->  0 U: [0,1) S: [0,1)
; CHECK-NEXT:   %v7 = sext i16 %v5 to i32
; CHECK-NEXT:   -->  {-1,+,-1}<nsw><%b1> U: [-256,0) S: [-256,0)  -->  -256 U: [-256,-255) S: [-256,-255)
; CHECK-NEXT: Determining loop execution counts for: @f0
; CHECK-NEXT: Loop %b1: backedge-taken count is 255
; CHECK-NEXT: Loop %b1: max backedge-taken count is 255
; CHECK-NEXT: Loop %b1: Predicated backedge-taken count is 255
; CHECK-NEXT:  Predicates:
; CHECK-EMPTY:
; CHECK-NEXT: Loop %b1: Trip multiple is 256


@g0 = global i32 0, align 4
@g1 = global i16 0, align 2

define signext i32 @f0() {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi i16 [ 0, %b0 ], [ %v2, %b1 ]
  %v2 = add i16 %v1, -1
  %v3 = mul i16 %v2, %v2
  %v4 = icmp eq i16 %v3, 0
  br i1 %v4, label %b2, label %b1

b2:                                               ; preds = %b1
  %v5 = phi i16 [ %v2, %b1 ]
  %v6 = phi i16 [ %v3, %b1 ]
  %v7 = sext i16 %v5 to i32
  store i32 %v7, i32* @g0, align 4
  store i16 %v6, i16* @g1, align 2
  ret i32 0
}

