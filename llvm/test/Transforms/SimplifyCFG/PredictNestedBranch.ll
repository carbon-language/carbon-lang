
; RUN: opt < %s -simplifycfg -dce -S | FileCheck %s

; Test that when == is true, all 6 comparisons evaluate to true or false
; ie, a == b implies a > b is false, but a >= b is true, and so on
define void @testEqTrue(i32 %a, i32 %b) {
; CHECK: @testEqTrue
; CHECK: icmp eq i32 %a, %b
; CHECK: call void @_Z1gi(i32 0)
; a == b implies a == b
; CHECK-NEXT: call void @_Z1gi(i32 1)
; a == b implies a >= b
; CHECK-NEXT: call void @_Z1gi(i32 3)
; a == b implies a <= b
; CHECK-NEXT: call void @_Z1gi(i32 4)
; CHECK: ret void
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end18

if.then:                                          ; preds = %entry
  call void @_Z1gi(i32 0)
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:                                         ; preds = %if.then
  call void @_Z1gi(i32 1)
  br label %if.end

if.end:                                           ; preds = %if.then2, %if.then
  %cmp3 = icmp ne i32 %a, %b
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:                                         ; preds = %if.end
  call void @_Z1gi(i32 2)
  br label %if.end5

if.end5:                                          ; preds = %if.then4, %if.end
  %cmp6 = icmp sge i32 %a, %b
  br i1 %cmp6, label %if.then7, label %if.end8

if.then7:                                         ; preds = %if.end5
  call void @_Z1gi(i32 3)
  br label %if.end8

if.end8:                                          ; preds = %if.then7, %if.end5
  %cmp9 = icmp sle i32 %a, %b
  br i1 %cmp9, label %if.then10, label %if.end11

if.then10:                                        ; preds = %if.end8
  call void @_Z1gi(i32 4)
  br label %if.end11

if.end11:                                         ; preds = %if.then10, %if.end8
  %cmp12 = icmp sgt i32 %a, %b
  br i1 %cmp12, label %if.then13, label %if.end14

if.then13:                                        ; preds = %if.end11
  call void @_Z1gi(i32 5)
  br label %if.end14

if.end14:                                         ; preds = %if.then13, %if.end11
  %cmp15 = icmp slt i32 %a, %b
  br i1 %cmp15, label %if.then16, label %if.end18

if.then16:                                        ; preds = %if.end14
  call void @_Z1gi(i32 6)
  br label %if.end18

if.end18:                                         ; preds = %if.end14, %if.then16, %entry
  ret void
}

; Test that when == is false, all 6 comparisons evaluate to true or false
; ie, a == b implies a > b is false, but a >= b is true, and so on
define void @testEqFalse(i32 %a, i32 %b) {
; CHECK: @testEqFalse
; CHECK: icmp eq i32 %a, %b
; CHECK: call void @_Z1gi(i32 0)
; CHECK-NOT: call void @_Z1gi(i32 1)
; CHECK-NOT: icmp ne
; CHECK: call void @_Z1gi(i32 2)
; CHECK: icmp sge
; CHECK: call void @_Z1gi(i32 3)
; CHECK: icmp sle
; CHECK: call void @_Z1gi(i32 4)
; CHECK: icmp sgt
; CHECK: call void @_Z1gi(i32 5)
; CHECK: icmp slt
; CHECK: call void @_Z1gi(i32 6)
; CHECK: ret void
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  call void @_Z1gi(i32 0)
  br label %if.end18
  
if.else:
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:                                         ; preds = %if.then
  call void @_Z1gi(i32 1)
  br label %if.end

if.end:                                           ; preds = %if.then2, %if.then
  %cmp3 = icmp ne i32 %a, %b
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:                                         ; preds = %if.end
  call void @_Z1gi(i32 2)
  br label %if.end5

if.end5:                                          ; preds = %if.then4, %if.end
  %cmp6 = icmp sge i32 %a, %b
  br i1 %cmp6, label %if.then7, label %if.end8

if.then7:                                         ; preds = %if.end5
  call void @_Z1gi(i32 3)
  br label %if.end8

if.end8:                                          ; preds = %if.then7, %if.end5
  %cmp9 = icmp sle i32 %a, %b
  br i1 %cmp9, label %if.then10, label %if.end11

if.then10:                                        ; preds = %if.end8
  call void @_Z1gi(i32 4)
  br label %if.end11

if.end11:                                         ; preds = %if.then10, %if.end8
  %cmp12 = icmp sgt i32 %a, %b
  br i1 %cmp12, label %if.then13, label %if.end14

if.then13:                                        ; preds = %if.end11
  call void @_Z1gi(i32 5)
  br label %if.end14

if.end14:                                         ; preds = %if.then13, %if.end11
  %cmp15 = icmp slt i32 %a, %b
  br i1 %cmp15, label %if.then16, label %if.end18

if.then16:                                        ; preds = %if.end14
  call void @_Z1gi(i32 6)
  br label %if.end18

if.end18:                                         ; preds = %if.end14, %if.then16, %entry
  ret void
}

declare void @_Z1gi(i32)
