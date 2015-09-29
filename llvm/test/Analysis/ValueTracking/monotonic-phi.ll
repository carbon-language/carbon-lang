; RUN: opt -instsimplify -S < %s | FileCheck %s

; CHECK-LABEL: @test1
define i1 @test1(i8 %p, i8* %pq, i8 %n, i8 %r) {
entry:
  br label %loop
loop:
  %A = phi i8 [ 1, %entry ], [ %next, %loop ]
  %next = add nsw i8 %A, 1
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %add = or i8 %A, %r
  %cmp = icmp eq i8 %add, 0
  ; CHECK: ret i1 false
  ret i1 %cmp
}

; CHECK-LABEL: @test2
define i1 @test2(i8 %p, i8* %pq, i8 %n, i8 %r) {
entry:
  br label %loop
loop:
  %A = phi i8 [ 1, %entry ], [ %next, %loop ]
  %next = add i8 %A, 1
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %add = or i8 %A, %r
  %cmp = icmp eq i8 %add, 0
  ; CHECK-NOT: ret i1 false
  ret i1 %cmp
}

; CHECK-LABEL: @test3
define i1 @test3(i8 %p, i8* %pq, i8 %n, i8 %r) {
entry:
  br label %loop
loop:
  %A = phi i8 [ 1, %entry ], [ %next, %loop ]
  %next = add nuw i8 %A, 1
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %add = or i8 %A, %r
  %cmp = icmp eq i8 %add, 0
  ; CHECK: ret i1 false
  ret i1 %cmp
}
