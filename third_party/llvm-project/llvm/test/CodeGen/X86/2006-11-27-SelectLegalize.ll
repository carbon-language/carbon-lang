; RUN: llc < %s -mtriple=i686-- | FileCheck %s
; PR1016

; CHECK: {{test.*1}}

define i32 @test(i32 %A, i32 %B, i32 %C) {
        %a = trunc i32 %A to i1         ; <i1> [#uses=1]
        %D = select i1 %a, i32 %B, i32 %C               ; <i32> [#uses=1]
        ret i32 %D
}

