; RUN: opt -S -basicaa -dse < %s | FileCheck %s

declare i8* @strcpy(i8* %dest, i8* %src) nounwind
define void @test1(i8* %src) {
; CHECK: @test1
  %B = alloca [16 x i8]
  %dest = getelementptr inbounds [16 x i8]* %B, i64 0, i64 0
; CHECK-NOT: @strcpy
  %call = call i8* @strcpy(i8* %dest, i8* %src)
; CHECK: ret void
  ret void
}

declare i8* @strncpy(i8* %dest, i8* %src, i32 %n) nounwind
define void @test2(i8* %src) {
; CHECK: @test2
  %B = alloca [16 x i8]
  %dest = getelementptr inbounds [16 x i8]* %B, i64 0, i64 0
; CHECK-NOT: @strcpy
  %call = call i8* @strncpy(i8* %dest, i8* %src, i32 12)
; CHECK: ret void
  ret void
}

declare i8* @strcat(i8* %dest, i8* %src) nounwind
define void @test3(i8* %src) {
; CHECK: @test3
  %B = alloca [16 x i8]
  %dest = getelementptr inbounds [16 x i8]* %B, i64 0, i64 0
; CHECK-NOT: @strcpy
  %call = call i8* @strcat(i8* %dest, i8* %src)
; CHECK: ret void
  ret void
}

declare i8* @strncat(i8* %dest, i8* %src, i32 %n) nounwind
define void @test4(i8* %src) {
; CHECK: @test4
  %B = alloca [16 x i8]
  %dest = getelementptr inbounds [16 x i8]* %B, i64 0, i64 0
; CHECK-NOT: @strcpy
  %call = call i8* @strncat(i8* %dest, i8* %src, i32 12)
; CHECK: ret void
  ret void
}

