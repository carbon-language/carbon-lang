; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)
declare void @foo(i8* nocapture)

define void @asan() sanitize_address {
entry:
  ; CHECK-LABEL: @asan(
  %text = alloca [1 x i8], align 1
  %0 = getelementptr inbounds [1 x i8], [1 x i8]* %text, i64 0, i64 0

  call void @llvm.lifetime.start(i64 1, i8* %0)
  call void @llvm.lifetime.end(i64 1, i8* %0)
  ; CHECK: call void @llvm.lifetime.start
  ; CHECK-NEXT: call void @llvm.lifetime.end

  call void @foo(i8* %0) ; Keep alloca alive

  ret void
}


define void @no_asan() {
entry:
  ; CHECK-LABEL: @no_asan(
  %text = alloca [1 x i8], align 1
  %0 = getelementptr inbounds [1 x i8], [1 x i8]* %text, i64 0, i64 0

  call void @llvm.lifetime.start(i64 1, i8* %0)
  call void @llvm.lifetime.end(i64 1, i8* %0)
  ; CHECK-NO: call void @llvm.lifetime

  call void @foo(i8* %0) ; Keep alloca alive

  ret void
}
