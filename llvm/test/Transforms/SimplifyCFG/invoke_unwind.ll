; This testcase checks to see if the simplifycfg pass is converting invoke
; instructions to call instructions if the handler just rethrows the exception.

; RUN: opt < %s -simplifycfg -S | FileCheck %s

declare void @bar()

define i32 @test1() {
; CHECK: @test1
; CHECK-NEXT: call void @bar()
; CHECK-NEXT: ret i32 0
        invoke void @bar( )
                        to label %Ok unwind label %Rethrow
Ok:             ; preds = %0
        ret i32 0
Rethrow:                ; preds = %0
        unwind
}
