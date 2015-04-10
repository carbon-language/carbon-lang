; RUN: opt -S -instcombine < %s | FileCheck %s
; <rdar://problem/8558713>

declare void @throwAnExceptionOrWhatever()

; CHECK-LABEL: @test1(
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
  tail call void @throwAnExceptionOrWhatever() nounwind
  br label %if.end

if.end:
; CHECK-NOT: trunc
  %conv9 = trunc i64 %add to i32
; CHECK: ret i32
  ret i32 %conv9
}

; CHECK-LABEL: @test2(
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
  tail call void @throwAnExceptionOrWhatever() nounwind
  br label %if.end

if.end:
  %conv9 = trunc i64 %add to i32
; CHECK: ret i32
  ret i32 %conv9
}

; CHECK: test3
; PR8816
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
  tail call void @throwAnExceptionOrWhatever() nounwind
  br label %if.end

if.end:
  ret i64 %add
; CHECK: ret i64
}

; CHECK-LABEL: @test4(
; Should be able to form an i8 sadd computed in an i32.
define zeroext i8 @test4(i8 signext %a, i8 signext %b) nounwind ssp {
entry:
  %conv = sext i8 %a to i32
  %conv2 = sext i8 %b to i32
  %add = add nsw i32 %conv2, %conv
  %add4 = add nsw i32 %add, 128
  %cmp = icmp ugt i32 %add4, 255
  br i1 %cmp, label %if.then, label %if.end
; CHECK: llvm.sadd.with.overflow.i8
if.then:                                          ; preds = %entry
  tail call void @throwAnExceptionOrWhatever() nounwind
  unreachable

if.end:                                           ; preds = %entry
  %conv7 = trunc i32 %add to i8
  ret i8 %conv7
; CHECK: ret i8
}

; CHECK-LABEL: @test8(
; PR11438
; This is @test1, but the operands are not sign-extended.  Make sure
; we don't transform this case.
define i32 @test8(i64 %a, i64 %b) nounwind ssp {
entry:
; CHECK-NOT: llvm.sadd
; CHECK: add i64 %a, %b
; CHECK-NOT: llvm.sadd
; CHECK: ret
  %add = add i64 %a, %b
  %add.off = add i64 %add, 2147483648
  %0 = icmp ugt i64 %add.off, 4294967295
  br i1 %0, label %if.then, label %if.end

if.then:
  tail call void @throwAnExceptionOrWhatever() nounwind
  br label %if.end

if.end:
  %conv9 = trunc i64 %add to i32
  ret i32 %conv9
}
