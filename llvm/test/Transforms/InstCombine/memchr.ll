; Test that the memchr library call simplifier works correctly.
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [14 x i8] c"hello world\5Cn\00"
@hellonull = constant [14 x i8] c"hello\00world\5Cn\00"
@null = constant [1 x i8] zeroinitializer
@chp = global i8* zeroinitializer

declare i8* @memchr(i8*, i32, i32)

define void @test1() {
; CHECK-LABEL: @test1
; CHECK: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @hello, i32 0, i32 6)
; CHECK-NOT: call i8* @memchr
; CHECK: ret void

  %str = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @memchr(i8* %str, i32 119, i32 14)
  store i8* %dst, i8** @chp
  ret void
}

define void @test2() {
; CHECK-LABEL: @test2
; CHECK: store i8* null, i8** @chp, align 4
; CHECK-NOT: call i8* @memchr
; CHECK: ret void

  %str = getelementptr [1 x i8], [1 x i8]* @null, i32 0, i32 0
  %dst = call i8* @memchr(i8* %str, i32 119, i32 1)
  store i8* %dst, i8** @chp
  ret void
}

define void @test3() {
; CHECK-LABEL: @test3
; CHECK: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @hello, i32 0, i32 13)
; CHECK-NOT: call i8* @memchr
; CHECK: ret void

  %src = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @memchr(i8* %src, i32 0, i32 14)
  store i8* %dst, i8** @chp
  ret void
}

define void @test4(i32 %chr) {
; CHECK-LABEL: @test4
; CHECK: call i8* @memchr
; CHECK-NOT: call i8* @memchr
; CHECK: ret void

  %src = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @memchr(i8* %src, i32 %chr, i32 14)
  store i8* %dst, i8** @chp
  ret void
}

define void @test5() {
; CHECK-LABEL: @test5
; CHECK: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @hello, i32 0, i32 13)
; CHECK-NOT: call i8* @memchr
; CHECK: ret void

  %src = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @memchr(i8* %src, i32 65280, i32 14)
  store i8* %dst, i8** @chp
  ret void
}

define void @test6() {
; CHECK-LABEL: @test6
; CHECK: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @hello, i32 0, i32 6)
; CHECK-NOT: call i8* @memchr
; CHECK: ret void

  %src = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
; Overflow, but we still find the right thing.
  %dst = call i8* @memchr(i8* %src, i32 119, i32 100)
  store i8* %dst, i8** @chp
  ret void
}

define void @test7() {
; CHECK-LABEL: @test7
; CHECK: store i8* null, i8** @chp, align 4
; CHECK-NOT: call i8* @memchr
; CHECK: ret void

  %src = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
; Overflow
  %dst = call i8* @memchr(i8* %src, i32 120, i32 100)
  store i8* %dst, i8** @chp
  ret void
}

define void @test8() {
; CHECK-LABEL: @test8
; CHECK: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @hellonull, i32 0, i32 6)
; CHECK-NOT: call i8* @memchr
; CHECK: ret void

  %str = getelementptr [14 x i8], [14 x i8]* @hellonull, i32 0, i32 0
  %dst = call i8* @memchr(i8* %str, i32 119, i32 14)
  store i8* %dst, i8** @chp
  ret void
}

define void @test9() {
; CHECK-LABEL: @test9
; CHECK: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @hellonull, i32 0, i32 6)
; CHECK-NOT: call i8* @memchr
; CHECK: ret void

  %str = getelementptr [14 x i8], [14 x i8]* @hellonull, i32 0, i32 2
  %dst = call i8* @memchr(i8* %str, i32 119, i32 12)
  store i8* %dst, i8** @chp
  ret void
}
