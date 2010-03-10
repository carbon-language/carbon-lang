; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s
; rdar://7610418

%ptr = type { i8* }
%struct.s1 = type { %ptr, %ptr }
%struct.s2 = type { i32, i8*, i8*, [256 x %struct.s1*], [8 x i32], i64, i8*, i32, i64, i64, i32, %struct.s3*, %struct.s3*, [49 x i64] }
%struct.s3 = type { %struct.s3*, %struct.s3*, i32, i32, i32 }

define fastcc i8* @t(i64 %size) nounwind {
entry:
; CHECK: t:
; CHECK: leaq (%rax,%rax,4)
  %0 = zext i32 undef to i64
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
