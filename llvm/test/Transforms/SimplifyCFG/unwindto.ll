; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | grep unwind_to | count 3

declare void @g(i32)

define i32 @f1() {
entry:
  br label %bb1
bb1: unwind_to %cleanup1
  call void @g(i32 0)
  br label %bb2
bb2: unwind_to %cleanup2
  call void @g(i32 1)
  br label %exit
exit:
  ret i32 0
cleanup1:
  ret i32 1
cleanup2:
  ret i32 2
}

define i32 @f2() {
entry: unwind_to %cleanup
  br label %bb1
bb1: unwind_to %cleanup
  br label %bb2
bb2: unwind_to %cleanup
  br label %bb3
bb3:
  br label %bb4
bb4: unwind_to %cleanup
  ret i32 0
cleanup:
  ret i32 1
}
