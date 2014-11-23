; RUN: opt < %s  -O3 -mcpu=corei7-avx -S | FileCheck %s -check-prefix=AVX1
; RUN: opt < %s  -O3 -mcpu=core-avx2 -S | FileCheck %s -check-prefix=AVX2
; RUN: opt < %s  -O3 -mcpu=knl -S | FileCheck %s -check-prefix=AVX512

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc_linux"

; The source code:
;
;void foo(int *A, int *B, int *trigger) {
;
;  for (int i=0; i<10000; i++) {
;    if (trigger[i] < 100) {
;          A[i] = B[i] + trigger[i];
;    }
;  }
;}


;AVX2: llvm.masked.load.v8i32
;AVX2: llvm.masked.store.v8i32
;AVX512: llvm.masked.load.v16i32
;AVX512: llvm.masked.store.v16i32
;AVX1-NOT: llvm.masked

; Function Attrs: nounwind uwtable
define void @foo(i32* %A, i32* %B, i32* %trigger) {
entry:
  %A.addr = alloca i32*, align 8
  %B.addr = alloca i32*, align 8
  %trigger.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store i32* %A, i32** %A.addr, align 8
  store i32* %B, i32** %B.addr, align 8
  store i32* %trigger, i32** %trigger.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4
  %cmp = icmp slt i32 %0, 10000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32* %i, align 4
  %idxprom = sext i32 %1 to i64
  %2 = load i32** %trigger.addr, align 8
  %arrayidx = getelementptr inbounds i32* %2, i64 %idxprom
  %3 = load i32* %arrayidx, align 4
  %cmp1 = icmp slt i32 %3, 100
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %4 = load i32* %i, align 4
  %idxprom2 = sext i32 %4 to i64
  %5 = load i32** %B.addr, align 8
  %arrayidx3 = getelementptr inbounds i32* %5, i64 %idxprom2
  %6 = load i32* %arrayidx3, align 4
  %7 = load i32* %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %8 = load i32** %trigger.addr, align 8
  %arrayidx5 = getelementptr inbounds i32* %8, i64 %idxprom4
  %9 = load i32* %arrayidx5, align 4
  %add = add nsw i32 %6, %9
  %10 = load i32* %i, align 4
  %idxprom6 = sext i32 %10 to i64
  %11 = load i32** %A.addr, align 8
  %arrayidx7 = getelementptr inbounds i32* %11, i64 %idxprom6
  store i32 %add, i32* %arrayidx7, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %12 = load i32* %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
