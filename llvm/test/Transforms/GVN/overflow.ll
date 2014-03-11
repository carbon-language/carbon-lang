; RUN: opt -S -gvn < %s | FileCheck %s

define i32 @sadd1(i32 %a, i32 %b) #0 {
; CHECK-LABEL: @sadd1(
entry:
  %sadd = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %cmp = extractvalue { i32, i1 } %sadd, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  ret i32 42

if.end:                                           ; preds = %entry
  %sadd3 = add i32 %a, %b
  ret i32 %sadd3
; CHECK-NOT: add i32 %a, %b
; CHECK: %sadd3.repl = extractvalue { i32, i1 } %sadd, 0
; CHECK: ret i32 %sadd3.repl
}

define i32 @sadd2(i32 %a, i32 %b) #0 {
entry:
  %sadd3 = add i32 %a, %b
  %sadd = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %cmp = extractvalue { i32, i1 } %sadd, 1
  br i1 %cmp, label %if.then, label %if.end
; CHECK-NOT: %sadd3 = add i32 %a, %b
; CHECK: %sadd.repl = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
; CHECK-NOT: %sadd = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
; CHECK: %sadd3.repl = extractvalue { i32, i1 } %sadd.repl, 0

if.then:                                          ; preds = %entry
  %sadd4 = add i32 %sadd3, 1
  ret i32 %sadd4
; CHECK: %sadd4 = add i32 %sadd3.repl, 1

if.end:                                           ; preds = %entry
  ret i32 %sadd3
; CHECK: ret i32 %sadd3.repl
}


declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone
