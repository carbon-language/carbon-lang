; RUN: opt < %s -simplifycfg -S | FileCheck %s

define void @test(i32* %P, i32* %Q, i1 %A, i1 %B) {
; CHECK: test
; CHECK: br i1
; CHECK-NOT: br i1
; CHECK: ret
; CHECK: ret

entry:
        br i1 %A, label %a, label %b
a:
        br i1 %B, label %b, label %c
b:
        store i32 123, i32* %P
        ret void
c:
        ret void
}
