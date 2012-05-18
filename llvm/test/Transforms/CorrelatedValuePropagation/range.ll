; RUN: opt -correlated-propagation -S < %s | FileCheck %s

declare i32 @foo()

define i32 @test1(i32 %a) nounwind {
  %a.off = add i32 %a, -8
  %cmp = icmp ult i32 %a.off, 8
  br i1 %cmp, label %then, label %else

then:
  %dead = icmp eq i32 %a, 7
  br i1 %dead, label %end, label %else

else:
  ret i32 1

end:
  ret i32 2

; CHECK: @test1
; CHECK: then:
; CHECK-NEXT: br i1 false, label %end, label %else
}

define i32 @test2(i32 %a) nounwind {
  %a.off = add i32 %a, -8
  %cmp = icmp ult i32 %a.off, 8
  br i1 %cmp, label %then, label %else

then:
  %dead = icmp ugt i32 %a, 15
  br i1 %dead, label %end, label %else

else:
  ret i32 1

end:
  ret i32 2

; CHECK: @test2
; CHECK: then:
; CHECK-NEXT: br i1 false, label %end, label %else
}

; CHECK: @test3
define i32 @test3(i32 %c) nounwind {
  %cmp = icmp slt i32 %c, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:
  ret i32 1

if.end:
  %cmp1 = icmp slt i32 %c, 3
  br i1 %cmp1, label %if.then2, label %if.end8

; CHECK: if.then2
if.then2:
  %cmp2 = icmp eq i32 %c, 2
; CHECK: br i1 true
  br i1 %cmp2, label %if.then4, label %if.end6

; CHECK: if.end6
if.end6:
  ret i32 2

if.then4:
  ret i32 3

if.end8:
  ret i32 4
}
