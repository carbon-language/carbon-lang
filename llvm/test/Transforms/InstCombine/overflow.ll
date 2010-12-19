; RUN: opt -S -instcombine < %s | FileCheck %s
; <rdar://problem/8558713>

declare i32 @throwAnExceptionOrWhatever(...)

; CHECK: @test1
define i32 @test1(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-NOT: sext
  %conv = sext i32 %a to i64
  %conv2 = sext i32 %b to i64
  %add = add nsw i64 %conv2, %conv
  %add.off = add i64 %add, 2147483648
; CHECK: llvm.sadd.with.overflow.i32
  %0 = icmp ugt i64 %add.off, 4294967295
  br i1 %0, label %if.then, label %if.end

if.then:
  %call = tail call i32 (...)* @throwAnExceptionOrWhatever() nounwind
  br label %if.end

if.end:
; CHECK-NOT: trunc
  %conv9 = trunc i64 %add to i32
; CHECK: ret i32
  ret i32 %conv9
}

; CHECK: @test2
; This form should not be promoted for two reasons: 1) it is unprofitable to
; promote it since the add.off instruction has another use, and 2) it is unsafe
; because the add-with-off makes the high bits of the original add live.
define i32 @test2(i32 %a, i32 %b, i64* %P) nounwind ssp {
entry:
  %conv = sext i32 %a to i64
  %conv2 = sext i32 %b to i64
  %add = add nsw i64 %conv2, %conv
  %add.off = add i64 %add, 2147483648
  
  store i64 %add.off, i64* %P
  
; CHECK-NOT: llvm.sadd.with.overflow
  %0 = icmp ugt i64 %add.off, 4294967295
  br i1 %0, label %if.then, label %if.end

if.then:
  %call = tail call i32 (...)* @throwAnExceptionOrWhatever() nounwind
  br label %if.end

if.end:
  %conv9 = trunc i64 %add to i32
; CHECK: ret i32
  ret i32 %conv9
}

; CHECK: test3
; This is illegal to transform because the high bits of the original add are
; live out.
define i64 @test3(i32 %a, i32 %b) nounwind ssp {
entry:
  %conv = sext i32 %a to i64
  %conv2 = sext i32 %b to i64
  %add = add nsw i64 %conv2, %conv
  %add.off = add i64 %add, 2147483648
; CHECK-NOT: llvm.sadd.with.overflow
  %0 = icmp ugt i64 %add.off, 4294967295
  br i1 %0, label %if.then, label %if.end

if.then:
  %call = tail call i32 (...)* @throwAnExceptionOrWhatever() nounwind
  br label %if.end

if.end:
  ret i64 %add
; CHECK: ret i64
}
