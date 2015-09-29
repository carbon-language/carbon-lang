; RUN: opt < %s -loop-unswitch -loop-unswitch-with-block-frequency -S 2>&1 | FileCheck %s

;; trivial condition should be unswithed regardless of coldness.
define i32 @test1(i1 %cond1, i1 %cond2) !prof !1 {
  br i1 %cond1, label %loop_begin, label %loop_exit, !prof !0

loop_begin:
; CHECK: br i1 true, label %continue, label %loop_exit.loopexit
  br i1 %cond2, label %continue, label %loop_exit  ; trivial condition

continue:
  call void @some_func1() noreturn nounwind
  br label %loop_begin

loop_exit:
  ret i32 0
}

;; cold non-trivial condition should not be unswitched.
define i32 @test2(i32* %var, i1 %cond1, i1 %cond2) !prof !1 {
  br i1 %cond1, label %loop_begin, label %loop_exit, !prof !0

loop_begin:
  store i32 1, i32* %var
; CHECK: br i1 %cond2, label %continue1, label %continue2
  br i1 %cond2, label %continue1, label %continue2  ; non-trivial condition

continue1:
  call void @some_func1() noreturn nounwind
  br label %joint

continue2:
  call void @some_func2() noreturn nounwind
  br label %joint

joint:
;; unswitching will duplicate these calls.
  call void @some_func3() noreturn nounwind
  call void @some_func4() noreturn nounwind
  br label %loop_begin

loop_exit:
  ret i32 0
}

declare void @some_func1() noreturn
declare void @some_func2() noreturn
declare void @some_func3() noreturn
declare void @some_func4() noreturn

!0 = !{!"branch_weights", i32 1, i32 100000000}
!1 = !{!"function_entry_count", i64 100}
