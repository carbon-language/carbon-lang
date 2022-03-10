; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; This testcase used to crash due to putting the bitsplit instruction in a
; wrong place.
; CHECK: bitsplit

target triple = "hexagon"

define fastcc i32 @fred(i32 %a0, i8* %a1, i1 %a2, i1 %a3) #0 {
b1:
  %v2 = lshr i32 %a0, 16
  %v3 = trunc i32 %v2 to i8
  br i1 %a2, label %b6, label %b4

b4:                                               ; preds = %b1
  %v5 = and i32 %a0, 65535
  br i1 %a3, label %b8, label %b9

b6:                                               ; preds = %b1
  %v7 = and i32 %a0, 65535
  br label %b9

b8:                                               ; preds = %b4
  store i8 %v3, i8* %a1, align 2
  ret i32 1

b9:                                               ; preds = %b6, %b4
  %v10 = phi i32 [ %v7, %b6 ], [ %v5, %b4 ]
  ret i32 %v10
}

attributes #0 = { nounwind optsize "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }
