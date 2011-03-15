; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s
; rdar://7610418

%ptr = type { i8* }
%struct.s1 = type { %ptr, %ptr }
%struct.s2 = type { i32, i8*, i8*, [256 x %struct.s1*], [8 x i32], i64, i8*, i32, i64, i64, i32, %struct.s3*, %struct.s3*, [49 x i64] }
%struct.s3 = type { %struct.s3*, %struct.s3*, i32, i32, i32 }

define fastcc i8* @t(i32 %base) nounwind {
entry:
; CHECK: t:
; CHECK: leaq (%rax,%rax,4)
  %0 = zext i32 %base to i64
  %1 = getelementptr inbounds %struct.s2* null, i64 %0
  br i1 undef, label %bb1, label %bb2

bb1:
; CHECK: %bb1
; CHECK-NOT: shlq $9
; CHECK-NOT: leaq
; CHECK: call
  %2 = getelementptr inbounds %struct.s2* null, i64 %0, i32 0
  call void @bar(i32* %2) nounwind
  unreachable

bb2:
; CHECK: %bb2
; CHECK-NOT: leaq
; CHECK: callq
  %3 = call fastcc i8* @foo(%struct.s2* %1) nounwind
  unreachable

bb3:
  ret i8* undef
}

declare void @bar(i32*)

declare fastcc i8* @foo(%struct.s2*) nounwind

; rdar://8773371

declare void @printf(...) nounwind

define void @commute(i32 %test_case, i32 %scale) nounwind ssp {
; CHECK: commute:
entry:
  switch i32 %test_case, label %sw.bb307 [
    i32 1, label %sw.bb
    i32 2, label %sw.bb
    i32 3, label %sw.bb
  ]

sw.bb:                                            ; preds = %entry, %entry, %entry
  %mul = mul nsw i32 %test_case, 3
  %mul20 = mul nsw i32 %mul, %scale
  br i1 undef, label %if.end34, label %sw.bb307

if.end34:                                         ; preds = %sw.bb
; CHECK: %if.end34
; CHECK: imull
; CHECK: leal
; CHECK-NOT: imull
  tail call void (...)* @printf(i32 %test_case, i32 %mul20) nounwind
  %tmp = mul i32 %scale, %test_case
  %tmp752 = mul i32 %tmp, 3
  %tmp753 = zext i32 %tmp752 to i64
  br label %bb.nph743.us

for.body53.us:                                    ; preds = %bb.nph743.us, %for.body53.us
  %exitcond = icmp eq i64 undef, %tmp753
  br i1 %exitcond, label %bb.nph743.us, label %for.body53.us

bb.nph743.us:                                     ; preds = %for.body53.us, %if.end34
  br label %for.body53.us

sw.bb307:                                         ; preds = %sw.bb, %entry
  ret void
}
