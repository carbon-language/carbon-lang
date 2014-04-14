; RUN: opt %loadPolly -polly-codegen -polly-vectorizer=polly < %s
; PR 19421
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define void @extract_field(i32* %frame, i32 %nb_planes) {
entry:
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
  %arrayidx2.moved.to.if.end = getelementptr i32* %frame, i64 %indvar
  %.moved.to.if.end = zext i32 %nb_planes to i64
  store i32 undef, i32* %arrayidx2.moved.to.if.end
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, %.moved.to.if.end
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}
