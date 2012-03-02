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
