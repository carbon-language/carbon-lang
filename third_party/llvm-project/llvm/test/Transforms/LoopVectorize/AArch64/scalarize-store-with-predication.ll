; RUN: opt -loop-vectorize -force-vector-width=1 -force-vector-interleave=2 -S -o - < %s | FileCheck %s
; RUN: opt -mattr=+sve -loop-vectorize -force-vector-width=1 -force-vector-interleave=2 -S -o - < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; This test is defending against a bug that appeared when we have a target
; configuration where masked loads/stores are legal -- e.g. AArch64 with SVE.
; Predication would not be applied during interleaving, enabling the
; possibility of superfluous loads/stores which could result in miscompiles.
; This test checks that, when we disable vectorisation and force interleaving,
; stores are predicated properly.
;
; This is _not_ an SVE-specific test. The same bug could manifest on any
; architecture with masked loads/stores, but we use SVE for testing purposes
; here.

define void @foo(i32* %data1, i32* %data2) {
; CHECK-LABEL: @foo(
; CHECK:       vector.body:
; CHECK:         br i1 {{%.*}}, label %pred.store.if, label %pred.store.continue
; CHECK:       pred.store.if:
; CHECK-NEXT:    store i32 {{%.*}}, i32* {{%.*}}
; CHECK-NEXT:    br label %pred.store.continue
; CHECK:       pred.store.continue:
; CHECK-NEXT:    br i1 {{%.*}}, label %pred.store.if2, label %pred.store.continue3
; CHECK:       pred.store.if2:
; CHECK-NEXT:    store i32 {{%.*}}, i32* {{%.*}}
; CHECK-NEXT:    br label %pred.store.continue3
; CHECK:       pred.store.continue3:

entry:
  br label %while.body

while.body:
  %i = phi i64 [ 1023, %entry ], [ %i.next, %if.end ]
  %arrayidx = getelementptr inbounds i32, i32* %data1, i64 %i
  %ld = load i32, i32* %arrayidx, align 4
  %cmp = icmp sgt i32 %ld, %ld
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %ld, i32* %arrayidx, align 4
  br label %if.end

if.end:
  %i.next = add nsw i64 %i, -1
  %tobool.not = icmp eq i64 %i, 0
  br i1 %tobool.not, label %while.end, label %while.body

while.end:
  ret void
}
