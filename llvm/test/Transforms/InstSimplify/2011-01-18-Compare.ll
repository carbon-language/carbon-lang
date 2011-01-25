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
  %r = lshr i32 %y, 1
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
