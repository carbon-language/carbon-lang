; RUN: llc < %s -enable-misched -march=thumb -mcpu=swift \
; RUN:          -pre-RA-sched=source -scheditins=false -ilp-window=0 \
; RUN:          -disable-ifcvt-triangle-false -disable-post-ra | FileCheck %s
;
; For these tests, we set -ilp-window=0 to simulate in order processor.

; %val1 is a 3-cycle load live out of %entry. It should be hoisted
; above the add.
; CHECK: @testload
; CHECK: %entry
; CHECK: ldr
; CHECK: adds
; CHECK: bne
; CHECK: %true
define i32 @testload(i32 *%ptr, i32 %sumin) {
entry:
  %sum1 = add i32 %sumin, 1
  %val1 = load i32* %ptr
  %p = icmp eq i32 %sumin, 0
  br i1 %p, label %true, label %end
true:
  %sum2 = add i32 %sum1, 1
  %ptr2 = getelementptr i32* %ptr, i32 1
  %val = load i32* %ptr2
  %val2 = add i32 %val1, %val
  br label %end
end:
  %valmerge = phi i32 [ %val1, %entry], [ %val2, %true ]
  %summerge = phi i32 [ %sum1, %entry], [ %sum2, %true ]
  %sumout = add i32 %valmerge, %summerge
  ret i32 %sumout
}

; The prefetch gets a default latency of 3 cycles and should be hoisted
; above the add.
;
; CHECK: @testprefetch
; CHECK: %entry
; CHECK: pld
; CHECK: adds
; CHECK: bx
define i32 @testprefetch(i8 *%ptr, i32 %i) {
entry:
  %tmp = add i32 %i, 1
  tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 3, i32 1 )
  ret i32 %tmp
}
declare void @llvm.prefetch(i8*, i32, i32, i32) nounwind
