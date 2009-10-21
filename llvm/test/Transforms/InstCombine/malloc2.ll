; RUN: opt < %s -instcombine -S | FileCheck %s
; PR1313

define i32 @test1(i32 %argc, i8* %argv, i8* %envp) {
        %tmp15.i.i.i23 = malloc [2564 x i32]            ; <[2564 x i32]*> [#uses=1]
; CHECK-NOT: call i8* @malloc
        %c = icmp eq [2564 x i32]* %tmp15.i.i.i23, null              ; <i1>:0 [#uses=1]
        %retval = zext i1 %c to i32             ; <i32> [#uses=1]
        ret i32 %retval
; CHECK: ret i32 0
}

define i32 @test2(i32 %argc, i8* %argv, i8* %envp) {
        %tmp15.i.i.i23 = malloc [2564 x i32]            ; <[2564 x i32]*> [#uses=1]
; CHECK-NOT: call i8* @malloc
        %X = bitcast [2564 x i32]* %tmp15.i.i.i23 to i32*
        %c = icmp ne i32* %X, null
        %retval = zext i1 %c to i32             ; <i32> [#uses=1]
        ret i32 %retval
; CHECK: ret i32 1
}

