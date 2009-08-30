; RUN: llvm-as < %s | opt -instcombine | llvm-dis | FileCheck %s

target datalayout = "e-p:64:64"

%pair = type { i32, i32 }
%struct.B = type { double }
%struct.A = type { %struct.B, i32, i32 }


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


; PR4748
define i32 @test12(%struct.A* %a) {
entry:
  %g3 = getelementptr %struct.A* %a, i32 0, i32 1
  store i32 10, i32* %g3, align 4

  %g4 = getelementptr %struct.A* %a, i32 0, i32 0
  
  %new_a = bitcast %struct.B* %g4 to %struct.A*

  %g5 = getelementptr %struct.A* %new_a, i32 0, i32 1	
  %a_a = load i32* %g5, align 4	
  ret i32 %a_a
; CHECK:      @test12
; CHECK:      getelementptr %struct.A* %a, i64 0, i32 1
; CHECK-NEXT: store i32 10, i32* %g3
; CHECK-NEXT: ret i32 10
}


; PR2235
%S = type { i32, [ 100 x i32] }
define i1 @test13(i64 %X, %S* %P) {
        %A = getelementptr inbounds %S* %P, i32 0, i32 1, i64 %X
        %B = getelementptr inbounds %S* %P, i32 0, i32 0
	%C = icmp eq i32* %A, %B
	ret i1 %C
; CHECK: @test13
; CHECK:    %C = icmp eq i64 %X, -1
}


@G = external global [3 x i8]           ; <[3 x i8]*> [#uses=1]
define i8* @test14(i32 %Idx) {
        %idx = zext i32 %Idx to i64             ; <i64> [#uses=1]
        %tmp = getelementptr i8* getelementptr ([3 x i8]* @G, i32 0, i32 0), i64 %idx
        ret i8* %tmp
; CHECK: @test14
; CHECK: getelementptr [3 x i8]* @G, i64 0, i64 %idx
}


; Test folding of constantexpr geps into normal geps.
@Array = external global [40 x i32]             ; <[40 x i32]*> [#uses=2]
define i32 *@test15(i64 %X) {
        %A = getelementptr i32* getelementptr ([40 x i32]* @Array, i64 0, i64 0), i64 %X
        ret i32* %A
; CHECK: @test15
; CHECK: getelementptr [40 x i32]* @Array, i64 0, i64 %X
}


define i32* @test16(i32* %X, i32 %Idx) {
        %R = getelementptr i32* %X, i32 %Idx       
        ret i32* %R
; CHECK: @test16
; CHECK: sext i32 %Idx to i64
}


define i1 @test17(i16* %P, i32 %I, i32 %J) {
        %X = getelementptr inbounds i16* %P, i32 %I
        %Y = getelementptr inbounds i16* %P, i32 %J
        %C = icmp ult i16* %X, %Y
        ret i1 %C
; CHECK: @test17
; CHECK: %C = icmp slt i32 %I, %J 
}

define i1 @test18(i16* %P, i32 %I) {
        %X = getelementptr inbounds i16* %P, i32 %I
        %C = icmp ult i16* %X, %P
        ret i1 %C
; CHECK: @test18
; CHECK: %C = icmp slt i32 %I, 0
}

define i32 @test19(i32* %P, i32 %A, i32 %B) {
        %tmp.4 = getelementptr inbounds i32* %P, i32 %A
        %tmp.9 = getelementptr inbounds i32* %P, i32 %B
        %tmp.10 = icmp eq i32* %tmp.4, %tmp.9
        %tmp.11 = zext i1 %tmp.10 to i32
        ret i32 %tmp.11
; CHECK: @test19
; CHECK: icmp eq i32 %A, %B
}

define i32 @test20(i32* %P, i32 %A, i32 %B) {
        %tmp.4 = getelementptr inbounds i32* %P, i32 %A
        %tmp.6 = icmp eq i32* %tmp.4, %P
        %tmp.7 = zext i1 %tmp.6 to i32
        ret i32 %tmp.7
; CHECK: @test20
; CHECK: icmp eq i32 %A, 0
}



