; RUN: opt < %s -bounds-checking -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare noalias i8* @strdup(i8* nocapture) nounwind
declare noalias i8* @strndup(i8* nocapture, i64) nounwind

; CHECK: @f1
define i8 @f1(i8* nocapture %str, i8** nocapture %esc) nounwind uwtable ssp {
; CHECK: call i64 @strlen(i8* %str)
; CHECK-NEXT: %1 = add nuw i64 {{.*}}, 1
  %call = tail call i8* @strdup(i8* %str) nounwind
  store i8* %call, i8** %esc, align 8
  %arrayidx = getelementptr inbounds i8* %call, i64 3
; CHECK: sub i64 %1, 3
  %1 = load i8* %arrayidx, align 1
  ret i8 %1
; CHECK: call void @llvm.trap
}

; CHECK: @f2
define i8 @f2(i8* nocapture %str, i8** nocapture %esc, i64 %limit) nounwind uwtable ssp {
; CHECK: call i64 @strnlen(i8* %str, i64 %limit)
; CHECK-NEXT: %1 = add nuw i64 {{.*}}, 1
  %call = tail call i8* @strndup(i8* %str, i64 %limit) nounwind
  store i8* %call, i8** %esc, align 8
  %arrayidx = getelementptr inbounds i8* %call, i64 3
; CHECK: sub i64 %1, 3
  %1 = load i8* %arrayidx, align 1
  ret i8 %1
; CHECK: call void @llvm.trap
}
