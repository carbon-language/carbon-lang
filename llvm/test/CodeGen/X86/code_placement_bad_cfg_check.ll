; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux < %s | FileCheck %s -check-prefix=CHECK-BAD-CFG
; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -no-bad-cfg-conflict-check < %s | FileCheck %s -check-prefix=CHECK-NO-BAD-CFG
; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -only-hot-bad-cfg-conflict-check < %s | FileCheck %s -check-prefix=CHECK-HOT-BAD-CFG

define void @foo(i32 %t) {
; Test that we lift the call to 'c' up to immediately follow the call to 'b'
; when we disable the cfg conflict check.
;
; CHECK-BAD-CFG-LABEL: foo:
; CHECK-BAD-CFG: callq b
; CHECK-BAD-CFG: callq a
; CHECK-BAD-CFG: callq c
;
; CHECK-NO-BAD-CFG-LABEL: foo:
; CHECK-NO-BAD-CFG: callq b
; CHECK-NO-BAD-CFG: callq c
; CHECK-NO-BAD-CFG: callq a
;
; CHECK-HOT-BAD-CFG-LABEL: foo:
; CHECK-HOT-BAD-CFG: callq b
; CHECK-HOT-BAD-CFG: callq c
; CHECK-HOT-BAD-CFG: callq a

entry:
  %cmp = icmp eq i32 %t, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  call void @a()
  br label %if.end

if.else:
  call void @b()
  br label %if.end

if.end:
  call void @c()
  ret void
}

define void @bar(i32 %t1, i32 %t2, i32 %t3) {
; Test that we lift the call to 'c' up to immediately follow the call to 'b'
; when we disable the cfg conflict check.
;
; CHECK-BAD-CFG-LABEL: bar:
; CHECK-BAD-CFG: callq a
; CHECK-BAD-CFG: callq c
; CHECK-BAD-CFG: callq d
; CHECK-BAD-CFG: callq f
; CHECK-BAD-CFG: callq b
; CHECK-BAD-CFG: callq e
; CHECK-BAD-CFG: callq g
;
; CHECK-NO-BAD-CFG-LABEL: bar:
; CHECK-NO-BAD-CFG: callq a
; CHECK-NO-BAD-CFG: callq c
; CHECK-NO-BAD-CFG: callq g
; CHECK-NO-BAD-CFG: callq d
; CHECK-NO-BAD-CFG: callq f
; CHECK-NO-BAD-CFG: callq b
; CHECK-NO-BAD-CFG: callq e
;
; CHECK-HOT-BAD-CFG-LABEL: bar:
; CHECK-HOT-BAD-CFG: callq a
; CHECK-HOT-BAD-CFG: callq c
; CHECK-HOT-BAD-CFG: callq d
; CHECK-HOT-BAD-CFG: callq f
; CHECK-HOT-BAD-CFG: callq g
; CHECK-HOT-BAD-CFG: callq b
; CHECK-HOT-BAD-CFG: callq e

entry:
  br i1 undef, label %if1.then, label %if1.else

if1.then:
  call void @a()
  %cmp2 = icmp eq i32 %t2, 0
  br i1 %cmp2, label %if2.then, label %if2.else

if2.then:
  call void @b()
  br label %if.end

if2.else:
  call void @c()
  br label %if.end

if1.else:
  call void @d()
  %cmp3 = icmp eq i32 %t3, 0
  br i1 %cmp3, label %if3.then, label %if3.else

if3.then:
  call void @e()
  br label %if.end

if3.else:
  call void @f()
  br label %if.end

if.end:
  call void @g()
  ret void
}

declare void @a()
declare void @b()
declare void @c()
declare void @d()
declare void @e()
declare void @f()
declare void @g()
