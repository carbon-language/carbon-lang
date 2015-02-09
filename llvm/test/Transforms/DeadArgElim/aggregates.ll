; RUN: opt -S -deadargelim %s | FileCheck %s

; Case 0: the basic example: an entire aggregate use is returned, but it's
; actually only used in ways we can eliminate. We gain benefit from analysing
; the "use" and applying its results to all sub-values.

; CHECK-LABEL: define internal void @agguse_dead()

define internal { i32, i32 } @agguse_dead() {
  ret { i32, i32 } { i32 0, i32 1 }
}

define internal { i32, i32 } @test_agguse_dead() {
  %val = call { i32, i32 } @agguse_dead()
  ret { i32, i32 } %val
}



; Case 1: an opaque use of the aggregate exists (in this case dead). Otherwise
; only one value is used, so function can be simplified.

; CHECK-LABEL: define internal i32 @rets_independent_if_agguse_dead()
; CHECK: [[RET:%.*]] = extractvalue { i32, i32 } { i32 0, i32 1 }, 1
; CHECK: ret i32 [[RET]]

define internal { i32, i32 } @rets_independent_if_agguse_dead() {
  ret { i32, i32 } { i32 0, i32 1 }
}

define internal { i32, i32 } @test_rets_independent_if_agguse_dead(i1 %tst) {
  %val = call { i32, i32 } @rets_independent_if_agguse_dead()
  br i1 %tst, label %use_1, label %use_aggregate

use_1:
  ; This use can be classified as applying only to ret 1.
  %val0 = extractvalue { i32, i32 } %val, 1
  call void @callee(i32 %val0)
  ret { i32, i32 } undef

use_aggregate:
  ; This use is assumed to apply to both 0 and 1.
  ret { i32, i32 } %val
}

; Case 2: an opaque use of the aggregate exists (in this case *live*). Other
; uses shouldn't matter.

; CHECK-LABEL: define internal { i32, i32 } @rets_live_agguse()
; CHECK: ret { i32, i32 } { i32 0, i32 1 }

define internal { i32, i32 } @rets_live_agguse() {
  ret { i32, i32} { i32 0, i32 1 }
}

define { i32, i32 } @test_rets_live_aggues(i1 %tst) {
  %val = call { i32, i32 } @rets_live_agguse()
  br i1 %tst, label %use_1, label %use_aggregate

use_1:
  ; This use can be classified as applying only to ret 1.
  %val0 = extractvalue { i32, i32 } %val, 1
  call void @callee(i32 %val0)
  ret { i32, i32 } undef

use_aggregate:
  ; This use is assumed to apply to both 0 and 1.
  ret { i32, i32 } %val
}

declare void @callee(i32)