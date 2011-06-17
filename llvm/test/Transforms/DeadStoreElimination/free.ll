; RUN: opt < %s -basicaa -dse -S | FileCheck %s

target datalayout = "e-p:64:64:64"

; CHECK: @test
; CHECK-NEXT: bitcast
; CHECK-NEXT: @free
; CHECK-NEXT: ret void
define void @test(i32* %Q, i32* %P) {
        %DEAD = load i32* %Q            ; <i32> [#uses=1]
        store i32 %DEAD, i32* %P
        %1 = bitcast i32* %P to i8*
        tail call void @free(i8* %1)
        ret void
}

; CHECK: @test2
; CHECK-NEXT: bitcast
; CHECK-NEXT: @free
; CHECK-NEXT: ret void
define void @test2({i32, i32}* %P) {
	%Q = getelementptr {i32, i32} *%P, i32 0, i32 1
	store i32 4, i32* %Q
        %1 = bitcast {i32, i32}* %P to i8*
        tail call void @free(i8* %1)
	ret void
}

; CHECK: @test4
; CHECK-NOT: store
; CHECK: ret void
define void @test4() {
  %m = call i8* @malloc(i64 24)
  store i8 0, i8* %m
  %m1 = getelementptr i8* %m, i64 1
  store i8 1, i8* %m1
  call void @free(i8* %m)
  ret void
}

declare void @free(i8*)
declare i8* @malloc(i64)
