; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s

define i32 @f1(i32 %t) nounwind {
; CHECK: f1
; CHECK-NOT: tst
; CHECK: ands
; CHECK: it ne
; CHECK: mvnne
  %and = and i32 %t, 256
  %tobool = icmp eq i32 %and, 0
  %retval.0 = select i1 %tobool, i32 0, i32 -26
  ret i32 %retval.0
}
