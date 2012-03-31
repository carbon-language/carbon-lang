; RUN: opt < %s -inline -inline-threshold=20 -S | FileCheck %s

define internal i32 @callee1(i32 %A, i32 %B) {
  %C = sdiv i32 %A, %B
  ret i32 %C
}

define i32 @caller1() {
; CHECK: define i32 @caller1
; CHECK-NEXT: ret i32 3

  %X = call i32 @callee1( i32 10, i32 3 )
  ret i32 %X
}

define i32 @caller2() {
; Check that we can constant-prop through instructions after inlining callee21
; to get constants in the inlined callsite to callee22.
; FIXME: Currently, the threshold is fixed at 20 because we don't perform
; *recursive* cost analysis to realize that the nested call site will definitely
; inline and be cheap. We should eventually do that and lower the threshold here
; to 1.
;
; CHECK: @caller2
; CHECK-NOT: call void @callee2
; CHECK: ret

  %x = call i32 @callee21(i32 42, i32 48)
  ret i32 %x
}

define i32 @callee21(i32 %x, i32 %y) {
  %sub = sub i32 %y, %x
  %result = call i32 @callee22(i32 %sub)
  ret i32 %result
}

declare i8* @getptr()

define i32 @callee22(i32 %x) {
  %icmp = icmp ugt i32 %x, 42
  br i1 %icmp, label %bb.true, label %bb.false
bb.true:
  ; This block musn't be counted in the inline cost.
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  %x4 = add i32 %x3, 1
  %x5 = add i32 %x4, 1
  %x6 = add i32 %x5, 1
  %x7 = add i32 %x6, 1
  %x8 = add i32 %x7, 1

  ret i32 %x8
bb.false:
  ret i32 %x
}

define i32 @caller3() {
; Check that even if the expensive path is hidden behind several basic blocks,
; it doesn't count toward the inline cost when constant-prop proves those paths
; dead.
;
; CHECK: @caller3
; CHECK-NOT: call
; CHECK: ret i32 6

entry:
  %x = call i32 @callee3(i32 42, i32 48)
  ret i32 %x
}

define i32 @callee3(i32 %x, i32 %y) {
  %sub = sub i32 %y, %x
  %icmp = icmp ugt i32 %sub, 42
  br i1 %icmp, label %bb.true, label %bb.false

bb.true:
  %icmp2 = icmp ult i32 %sub, 64
  br i1 %icmp2, label %bb.true.true, label %bb.true.false

bb.true.true:
  ; This block musn't be counted in the inline cost.
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  %x4 = add i32 %x3, 1
  %x5 = add i32 %x4, 1
  %x6 = add i32 %x5, 1
  %x7 = add i32 %x6, 1
  %x8 = add i32 %x7, 1
  br label %bb.merge

bb.true.false:
  ; This block musn't be counted in the inline cost.
  %y1 = add i32 %y, 1
  %y2 = add i32 %y1, 1
  %y3 = add i32 %y2, 1
  %y4 = add i32 %y3, 1
  %y5 = add i32 %y4, 1
  %y6 = add i32 %y5, 1
  %y7 = add i32 %y6, 1
  %y8 = add i32 %y7, 1
  br label %bb.merge

bb.merge:
  %result = phi i32 [ %x8, %bb.true.true ], [ %y8, %bb.true.false ]
  ret i32 %result

bb.false:
  ret i32 %sub
}
