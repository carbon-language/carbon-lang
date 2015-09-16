; RUN: opt -instcombine -S < %s | FileCheck %s
; PR5438

; TODO: This should also optimize down.
;define i32 @test1(i32 %a, i32 %b) nounwind readnone {
;entry:
;        %0 = icmp sgt i32 %a, -1        ; <i1> [#uses=1]
;        %1 = icmp slt i32 %b, 0         ; <i1> [#uses=1]
;        %2 = xor i1 %1, %0              ; <i1> [#uses=1]
;        %3 = zext i1 %2 to i32          ; <i32> [#uses=1]
;        ret i32 %3
;}

; TODO: This optimizes partially but not all the way.
;define i32 @test2(i32 %a, i32 %b) nounwind readnone {
;entry:
;        %0 = and i32 %a, 8            ;<i32>  [#uses=1]
;        %1 = and i32 %b, 8            ;<i32>  [#uses=1]
;        %2 = icmp eq i32 %0, %1         ;<i1>  [#uses=1]
;        %3 = zext i1 %2 to i32          ;<i32>  [#uses=1]
;        ret i32 %3
;}

define i32 @test3(i32 %a, i32 %b) nounwind readnone {
; CHECK-LABEL: @test3(
entry:
; CHECK: [[XOR1:%.*]] = xor i32 %a, %b
; CHECK: [[SHIFT:%.*]] = lshr i32 [[XOR1]], 31
; CHECK: [[XOR2:%.*]] = xor i32 [[SHIFT]], 1
        %0 = lshr i32 %a, 31            ; <i32> [#uses=1]
        %1 = lshr i32 %b, 31            ; <i32> [#uses=1]
        %2 = icmp eq i32 %0, %1         ; <i1> [#uses=1]
        %3 = zext i1 %2 to i32          ; <i32> [#uses=1]
        ret i32 %3
; CHECK-NOT: icmp
; CHECK-NOT: zext
; CHECK: ret i32 [[XOR2]]
}

; Variation on @test3: checking the 2nd bit in a situation where the 5th bit
; is one, not zero.
define i32 @test3i(i32 %a, i32 %b) nounwind readnone {
; CHECK-LABEL: @test3i(
entry:
; CHECK: xor i32 %a, %b
; CHECK: lshr i32 %0, 31
; CHECK: xor i32 %1, 1
        %0 = lshr i32 %a, 29            ; <i32> [#uses=1]
        %1 = lshr i32 %b, 29            ; <i32> [#uses=1]
        %2 = or i32 %0, 35
        %3 = or i32 %1, 35
        %4 = icmp eq i32 %2, %3         ; <i1> [#uses=1]
        %5 = zext i1 %4 to i32          ; <i32> [#uses=1]
        ret i32 %5
; CHECK-NOT: icmp
; CHECK-NOT: zext
; CHECK: ret i32 %2
}

define i1 @test4a(i32 %a) {
; CHECK-LABEL: @test4a(
 entry:
; CHECK: %c = icmp slt i32 %a, 1
; CHECK-NEXT: ret i1 %c
  %l = ashr i32 %a, 31
  %na = sub i32 0, %a
  %r = lshr i32 %na, 31
  %signum = or i32 %l, %r
  %c = icmp slt i32 %signum, 1
  ret i1 %c
}

define i1 @test4b(i64 %a) {
; CHECK-LABEL: @test4b(
 entry:
; CHECK: %c = icmp slt i64 %a, 1
; CHECK-NEXT: ret i1 %c
  %l = ashr i64 %a, 63
  %na = sub i64 0, %a
  %r = lshr i64 %na, 63
  %signum = or i64 %l, %r
  %c = icmp slt i64 %signum, 1
  ret i1 %c
}

define i1 @test4c(i64 %a) {
; CHECK-LABEL: @test4c(
 entry:
; CHECK: %c = icmp slt i64 %a, 1
; CHECK-NEXT: ret i1 %c
  %l = ashr i64 %a, 63
  %na = sub i64 0, %a
  %r = lshr i64 %na, 63
  %signum = or i64 %l, %r
  %signum.trunc = trunc i64 %signum to i32
  %c = icmp slt i32 %signum.trunc, 1
  ret i1 %c
}
