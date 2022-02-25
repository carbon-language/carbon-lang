; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define i32 @test1(i32 %X) {
; CHECK: lsr{{.*}}#31
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
}

