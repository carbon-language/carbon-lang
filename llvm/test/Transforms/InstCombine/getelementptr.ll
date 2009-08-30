; RUN: llvm-as < %s | opt -instcombine | llvm-dis | FileCheck %s

%pair = type { i32, i32 }
@Global = constant [10 x i8] c"helloworld"

; Test noop elimination
define i32* @test1(i32* %I) {
        %A = getelementptr i32* %I, i64 0 
        ret i32* %A
; CHECK: @test1
; CHECK: ret i32* %I
}

; Test noop elimination
define i32* @test2(i32* %I) {
        %A = getelementptr i32* %I
        ret i32* %A
; CHECK: @test2
; CHECK: ret i32* %I
}

; Test that two array indexing geps fold
define i32* @test3(i32* %I) {
        %A = getelementptr i32* %I, i64 17
        %B = getelementptr i32* %A, i64 4
        ret i32* %B
; CHECK: @test3
; CHECK: getelementptr i32* %I, i64 21
}

; Test that two getelementptr insts fold
define i32* @test4({ i32 }* %I) {
        %A = getelementptr { i32 }* %I, i64 1 
        %B = getelementptr { i32 }* %A, i64 0, i32 0
        ret i32* %B
; CHECK: @test4
; CHECK: getelementptr %0* %I, i64 1, i32 0
}

define void @test5(i8 %B) {
        ; This should be turned into a constexpr instead of being an instruction
        %A = getelementptr [10 x i8]* @Global, i64 0, i64 4 
        store i8 %B, i8* %A
        ret void
; CHECK: @test5
; CHECK: store i8 %B, i8* getelementptr inbounds ([10 x i8]* @Global, i64 0, i64 4)
}

define i32* @test6() {
        %M = malloc [4 x i32] 
        %A = getelementptr [4 x i32]* %M, i64 0, i64 0
        %B = getelementptr i32* %A, i64 2             
        ret i32* %B
; CHECK: @test6
; CHECK: getelementptr [4 x i32]* %M, i64 0, i64 2
}

define i32* @test7(i32* %I, i64 %C, i64 %D) {
        %A = getelementptr i32* %I, i64 %C              ; <i32*> [#uses=1]
        %B = getelementptr i32* %A, i64 %D              ; <i32*> [#uses=1]
        ret i32* %B
; CHECK: @test7
; CHECK: %A.sum = add i64 %C, %D
; CHECK: getelementptr i32* %I, i64 %A.sum
}

define i8* @test8([10 x i32]* %X) {
        ;; Fold into the cast.
        %A = getelementptr [10 x i32]* %X, i64 0, i64 0         ; <i32*> [#uses=1]
        %B = bitcast i32* %A to i8*             ; <i8*> [#uses=1]
        ret i8* %B
; CHECK: @test8
; CHECK: bitcast [10 x i32]* %X to i8*
}

define i32 @test9() {
        %A = getelementptr { i32, double }* null, i32 0, i32 1          ; <double*> [#uses=1]
        %B = ptrtoint double* %A to i32         ; <i32> [#uses=1]
        ret i32 %B
; CHECK: @test9
; CHECK: ret i32 8
}

define i1 @test10({ i32, i32 }* %x, { i32, i32 }* %y) {
        %tmp.1 = getelementptr { i32, i32 }* %x, i32 0, i32 1           ; <i32*> [#uses=1]
        %tmp.3 = getelementptr { i32, i32 }* %y, i32 0, i32 1           ; <i32*> [#uses=1]
        ;; seteq x, y
        %tmp.4 = icmp eq i32* %tmp.1, %tmp.3            ; <i1> [#uses=1]
        ret i1 %tmp.4
; CHECK: @test10
; CHECK: icmp eq %pair* %x, %y
}

define i1 @test11({ i32, i32 }* %X) {
        %P = getelementptr { i32, i32 }* %X, i32 0, i32 0               ; <i32*> [#uses=1]
        %Q = icmp eq i32* %P, null              ; <i1> [#uses=1]
        ret i1 %Q
; CHECK: @test11
; CHECK: icmp eq %pair* %X, null
}
