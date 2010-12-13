; RUN: opt < %s -simplifycfg -S | FileCheck %s

declare void @bar()

; This testcase checks to see if the simplifycfg pass is converting invoke
; instructions to call instructions if the handler just rethrows the exception.
define i32 @test1() {
; CHECK: @test1
; CHECK-NEXT: call void @bar()
; CHECK-NEXT: ret i32 0
        invoke void @bar( )
                        to label %1 unwind label %Rethrow
        ret i32 0
Rethrow:
        unwind
}


; Verify that simplifycfg isn't duplicating 'unwind' instructions.  Doing this
; is bad because it discourages commoning.
define i32 @test2(i1 %c) {
; CHECK: @test2
; CHECK: T:
; CHECK-NEXT: call void @bar()
; CHECK-NEXT: br label %F
  br i1 %c, label %T, label %F
T:
  call void @bar()
  br label %F
F:
  unwind
}
