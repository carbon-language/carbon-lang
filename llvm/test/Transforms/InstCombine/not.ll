; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep xor

define i32 @test1(i32 %A) {
        %B = xor i32 %A, -1             ; <i32> [#uses=1]
        %C = xor i32 %B, -1             ; <i32> [#uses=1]
        ret i32 %C
}

define i1 @test2(i32 %A, i32 %B) {
        ; Can change into setge
        %cond = icmp sle i32 %A, %B             ; <i1> [#uses=1]
        %Ret = xor i1 %cond, true               ; <i1> [#uses=1]
        ret i1 %Ret
}

; Test that demorgans law can be instcombined
define i32 @test3(i32 %A, i32 %B) {
        %a = xor i32 %A, -1             ; <i32> [#uses=1]
        %b = xor i32 %B, -1             ; <i32> [#uses=1]
        %c = and i32 %a, %b             ; <i32> [#uses=1]
        %d = xor i32 %c, -1             ; <i32> [#uses=1]
        ret i32 %d
}

; Test that demorgens law can work with constants
define i32 @test4(i32 %A, i32 %B) {
        %a = xor i32 %A, -1             ; <i32> [#uses=1]
        %c = and i32 %a, 5              ; <i32> [#uses=1]
        %d = xor i32 %c, -1             ; <i32> [#uses=1]
        ret i32 %d
}

; test the mirror of demorgans law...
define i32 @test5(i32 %A, i32 %B) {
        %a = xor i32 %A, -1             ; <i32> [#uses=1]
        %b = xor i32 %B, -1             ; <i32> [#uses=1]
        %c = or i32 %a, %b              ; <i32> [#uses=1]
        %d = xor i32 %c, -1             ; <i32> [#uses=1]
        ret i32 %d
}

