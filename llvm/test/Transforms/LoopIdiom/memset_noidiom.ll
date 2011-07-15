; RUN: opt -basicaa -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; CHECK: @memset
; CHECK-NOT: callq _memset
define i8* @memset(i8* %b, i32 %c, i64 %len) nounwind uwtable ssp {
entry:
  %b.addr = alloca i8*, align 8
  %c.addr = alloca i32, align 4
  %len.addr = alloca i64, align 8
  %p = alloca i8*, align 8
  %i = alloca i32, align 4
  store i8* %b, i8** %b.addr, align 8
  store i32 %c, i32* %c.addr, align 4
  store i64 %len, i64* %len.addr, align 8
  %tmp = load i8** %b.addr, align 8
  store i8* %tmp, i8** %p, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %tmp2 = load i32* %i, align 4
  %conv = sext i32 %tmp2 to i64
  %tmp3 = load i64* %len.addr, align 8
  %cmp = icmp ult i64 %conv, %tmp3
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp5 = load i32* %c.addr, align 4
  %conv6 = trunc i32 %tmp5 to i8
  %tmp7 = load i8** %p, align 8
  %incdec.ptr = getelementptr inbounds i8* %tmp7, i32 1
  store i8* %incdec.ptr, i8** %p, align 8
  store i8 %conv6, i8* %tmp7
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %tmp8 = load i32* %i, align 4
  %inc = add nsw i32 %tmp8, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %tmp9 = load i8** %b.addr, align 8
  ret i8* %tmp9
}
