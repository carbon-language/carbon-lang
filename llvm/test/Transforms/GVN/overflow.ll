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

; Check if PRE does not crash
define i32 @pre(i32 %a, i32 %b) nounwind ssp uwtable {
entry:
  %cmp = icmp sgt i32 %a, 42
  br i1 %cmp, label %if.then, label %if.end3

if.then:                                          ; preds = %entry
  %add = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %add1 = extractvalue {i32, i1} %add, 0
  %o = extractvalue {i32, i1} %add, 1
  %o32 = zext i1 %o to i32
  %add32 = add i32 %add1, %o32
  %cmp1 = icmp sgt i32 %add1, 42
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:                                         ; preds = %if.then
  call void @abort() noreturn
  unreachable

if.end3:                                          ; preds = %if.end, %entry
  %add4 = add i32 %a, %b
  ret i32 %add4
}

declare void @abort() noreturn
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone
