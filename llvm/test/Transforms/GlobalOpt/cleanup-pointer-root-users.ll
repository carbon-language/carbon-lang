; RUN: opt -globalopt -S -o - < %s | FileCheck %s

@test1 = internal global i8* null

define void @test1a() {
; CHECK: @test1a
; CHECK-NOT: store
; CHECK-NEXT: ret void
  store i8* null, i8** @test1
  ret void
}

define void @test1b(i8* %p) {
; CHECK: @test1b
; CHECK-NEXT: store
; CHECK-NEXT: ret void
  store i8* %p, i8** @test1
  ret void
}
