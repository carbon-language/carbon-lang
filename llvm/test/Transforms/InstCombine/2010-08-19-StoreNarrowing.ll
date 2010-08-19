; RUN: opt -S -instcombine %s | not grep and
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%class.A = type { i8, [3 x i8] }

define void @_ZN1AC2Ev(%class.A* %this) nounwind ssp align 2 {
entry:
  %0 = bitcast %class.A* %this to i32*            ; <i32*> [#uses=5]
  %1 = load i32* %0, align 4                      ; <i32> [#uses=1]
  %2 = and i32 %1, -8                             ; <i32> [#uses=2]
  store i32 %2, i32* %0, align 4
  %3 = and i32 %2, -57                            ; <i32> [#uses=1]
  %4 = or i32 %3, 8                               ; <i32> [#uses=2]
  store i32 %4, i32* %0, align 4
  %5 = and i32 %4, -65                            ; <i32> [#uses=2]
  store i32 %5, i32* %0, align 4
  %6 = and i32 %5, -129                           ; <i32> [#uses=1]
  store i32 %6, i32* %0, align 4
  ret void
}
