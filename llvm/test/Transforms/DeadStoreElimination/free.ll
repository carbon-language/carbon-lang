; RUN: opt < %s -basicaa -dse -S | FileCheck %s

target datalayout = "e-p:64:64:64"

; CHECK: @test
; CHECK-NEXT: bitcast
; CHECK-NEXT: @free
; CHECK-NEXT: ret void
define void @test(i32* %Q, i32* %P) {
        %DEAD = load i32* %Q            ; <i32> [#uses=1]
        store i32 %DEAD, i32* %P
        free i32* %P
        ret void
}

; CHECK: @test2
; CHECK-NEXT: bitcast
; CHECK-NEXT: @free
; CHECK-NEXT: ret void
define void @test2({i32, i32}* %P) {
	%Q = getelementptr {i32, i32} *%P, i32 0, i32 1
	store i32 4, i32* %Q
	free {i32,i32}* %P
	ret void
}

