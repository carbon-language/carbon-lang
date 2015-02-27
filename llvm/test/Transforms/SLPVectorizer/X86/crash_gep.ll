; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-unknown-linux-gnu

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = common global i64* null, align 8

; Function Attrs: nounwind uwtable
define i32 @fn1() {
entry:
  %0 = load i64*, i64** @a, align 8
  %add.ptr = getelementptr inbounds i64, i64* %0, i64 1
  %1 = ptrtoint i64* %add.ptr to i64
  %arrayidx = getelementptr inbounds i64, i64* %0, i64 2
  store i64 %1, i64* %arrayidx, align 8
  %2 = ptrtoint i64* %arrayidx to i64
  store i64 %2, i64* %add.ptr, align 8
  ret i32 undef
}
