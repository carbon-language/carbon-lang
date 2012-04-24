; RUN: opt < %s -basicaa -gvn -instcombine -S | FileCheck %s

declare i32* @test(i32* nocapture)

define i32 @test2() {
; CHECK: ret i32 0
       %P = alloca i32
       %Q = call i32* @test(i32* %P)
       %a = load i32* %P
       store i32 4, i32* %Q   ;; cannot clobber P since it is nocapture.
       %b = load i32* %P
       %c = sub i32 %a, %b
       ret i32 %c
}

