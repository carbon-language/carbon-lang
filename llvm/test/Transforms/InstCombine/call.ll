; Ignore stderr, we expect warnings there
; RUN: opt < %s -instcombine 2> /dev/null -S | FileCheck %s

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; Simple case, argument translatable without changing the value
declare void @test1a(i8*)

define void @test1(i32* %A) {
        call void bitcast (void (i8*)* @test1a to void (i32*)*)( i32* %A )
        ret void
; CHECK: %tmp = bitcast i32* %A to i8*
; CHECK: call void @test1a(i8* %tmp)
; CHECK: ret void
}

; More complex case, translate argument because of resolution.  This is safe 
; because we have the body of the function
define void @test2a(i8 %A) {
        ret void
; CHECK: ret void
}

define i32 @test2(i32 %A) {
        call void bitcast (void (i8)* @test2a to void (i32)*)( i32 %A )
        ret i32 %A
; CHECK: %tmp = trunc i32 %A to i8
; CHECK: call void @test2a(i8 %tmp)
; CHECK: ret i32 %A
}


; Resolving this should insert a cast from sbyte to int, following the C 
; promotion rules.
declare void @test3a(i8, ...)

define void @test3(i8 %A, i8 %B) {
        call void bitcast (void (i8, ...)* @test3a to void (i8, i8)*)( i8 %A, i8 %B 
)
        ret void
; CHECK: %tmp = zext i8 %B to i32
; CHECK: call void (i8, ...)* @test3a(i8 %A, i32 %tmp)
; CHECK: ret void
}


; test conversion of return value...
define i8 @test4a() {
        ret i8 0
; CHECK: ret i8 0
}

define i32 @test4() {
        %X = call i32 bitcast (i8 ()* @test4a to i32 ()*)( )            ; <i32> [#uses=1]
        ret i32 %X
; CHECK: %X1 = call i8 @test4a()
; CHECK: %tmp = zext i8 %X1 to i32
; CHECK: ret i32 %tmp
}


; test conversion of return value... no value conversion occurs so we can do 
; this with just a prototype...
declare i32 @test5a()

define i32 @test5() {
        %X = call i32 @test5a( )                ; <i32> [#uses=1]
        ret i32 %X
; CHECK: %X = call i32 @test5a()
; CHECK: ret i32 %X
}


; test addition of new arguments...
declare i32 @test6a(i32)

define i32 @test6() {
        %X = call i32 bitcast (i32 (i32)* @test6a to i32 ()*)( )                ; <i32> [#uses=1]
        ret i32 %X
; CHECK: %X1 = call i32 @test6a(i32 0)
; CHECK: ret i32 %X1
}


; test removal of arguments, only can happen with a function body
define void @test7a() {
        ret void
; CHECK: ret void
}

define void @test7() {
        call void bitcast (void ()* @test7a to void (i32)*)( i32 5 )
        ret void
; CHECK: call void @test7a()
; CHECK: ret void
}


