; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @foo(i32* %P) #0 {
entry:
  %tobool = icmp eq i32* %P, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  store i32 0, i32* %P, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void

; CHECK: @foo
; CHECK: beqlr
; CHECK: blr
}

define void @bar(i32* %P, i32* %Q) #0 {
entry:
  %tobool = icmp eq i32* %P, null
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  store i32 0, i32* %P, align 4
  %tobool1 = icmp eq i32* %Q, null
  br i1 %tobool1, label %if.end3, label %if.then2

if.then2:                                         ; preds = %if.then
  store i32 1, i32* %Q, align 4
  br label %if.end3

if.else:                                          ; preds = %entry
  store i32 0, i32* %Q, align 4
  br label %if.end3

if.end3:                                          ; preds = %if.then, %if.then2, %if.else
  ret void

; CHECK: @bar
; CHECK: beqlr
; CHECK: blr
}

attributes #0 = { nounwind }
