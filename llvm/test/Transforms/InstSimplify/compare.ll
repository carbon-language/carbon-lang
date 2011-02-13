; RUN: opt < %s -instsimplify -S | FileCheck %s
target datalayout = "p:32:32"

define i1 @ptrtoint() {
; CHECK: @ptrtoint
  %a = alloca i8
  %tmp = ptrtoint i8* %a to i32
  %r = icmp eq i32 %tmp, 0
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @zext(i32 %x) {
; CHECK: @zext
  %e1 = zext i32 %x to i64
  %e2 = zext i32 %x to i64
  %r = icmp eq i64 %e1, %e2
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @zext2(i1 %x) {
; CHECK: @zext2
  %e = zext i1 %x to i32
  %c = icmp ne i32 %e, 0
  ret i1 %c
; CHECK: ret i1 %x
}

define i1 @zext3() {
; CHECK: @zext3
  %e = zext i1 1 to i32
  %c = icmp ne i32 %e, 0
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @sext(i32 %x) {
; CHECK: @sext
  %e1 = sext i32 %x to i64
  %e2 = sext i32 %x to i64
  %r = icmp eq i64 %e1, %e2
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @sext2(i1 %x) {
; CHECK: @sext2
  %e = sext i1 %x to i32
  %c = icmp ne i32 %e, 0
  ret i1 %c
; CHECK: ret i1 %x
}

define i1 @sext3() {
; CHECK: @sext3
  %e = sext i1 1 to i32
  %c = icmp ne i32 %e, 0
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @add(i32 %x, i32 %y) {
; CHECK: @add
  %l = lshr i32 %x, 1
  %q = lshr i32 %y, 1
  %r = or i32 %q, 1
  %s = add i32 %l, %r
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @add2(i8 %x, i8 %y) {
; CHECK: @add2
  %l = or i8 %x, 128
  %r = or i8 %y, 129
  %s = add i8 %l, %r
  %c = icmp eq i8 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @add3(i8 %x, i8 %y) {
; CHECK: @add3
  %l = zext i8 %x to i32
  %r = zext i8 %y to i32
  %s = add i32 %l, %r
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 %c
}

define i1 @add4(i32 %x, i32 %y) {
; CHECK: @add4
  %z = add nsw i32 %y, 1
  %s1 = add nsw i32 %x, %y
  %s2 = add nsw i32 %x, %z
  %c = icmp slt i32 %s1, %s2
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @add5(i32 %x, i32 %y) {
; CHECK: @add5
  %z = add nuw i32 %y, 1
  %s1 = add nuw i32 %x, %z
  %s2 = add nuw i32 %x, %y
  %c = icmp ugt i32 %s1, %s2
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @addpowtwo(i32 %x, i32 %y) {
; CHECK: @addpowtwo
  %l = lshr i32 %x, 1
  %r = shl i32 1, %y
  %s = add i32 %l, %r
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @or(i32 %x) {
; CHECK: @or
  %o = or i32 %x, 1
  %c = icmp eq i32 %o, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @shl(i32 %x) {
; CHECK: @shl
  %s = shl i32 1, %x
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @lshr(i32 %x) {
; CHECK: @lshr
  %s = lshr i32 -1, %x
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @ashr(i32 %x) {
; CHECK: @ashr
  %s = ashr i32 -1, %x
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @select1(i1 %cond) {
; CHECK: @select1
  %s = select i1 %cond, i32 1, i32 0
  %c = icmp eq i32 %s, 1
  ret i1 %c
; CHECK: ret i1 %cond
}

define i1 @select2(i1 %cond) {
; CHECK: @select2
  %x = zext i1 %cond to i32
  %s = select i1 %cond, i32 %x, i32 0
  %c = icmp ne i32 %s, 0
  ret i1 %c
; CHECK: ret i1 %cond
}

define i1 @select3(i1 %cond) {
; CHECK: @select3
  %x = zext i1 %cond to i32
  %s = select i1 %cond, i32 1, i32 %x
  %c = icmp ne i32 %s, 0
  ret i1 %c
; CHECK: ret i1 %cond
}

define i1 @select4(i1 %cond) {
; CHECK: @select4
  %invert = xor i1 %cond, 1
  %s = select i1 %invert, i32 0, i32 1
  %c = icmp ne i32 %s, 0
  ret i1 %c
; CHECK: ret i1 %cond
}
