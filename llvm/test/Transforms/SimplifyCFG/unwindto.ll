; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | grep {unwinds to} | count 3

declare void @g(i32)

define i32 @f1() {
entry:
  br label %bb1
bb1: unwinds to %cleanup1
  call void @g(i32 0)
  br label %bb2
bb2: unwinds to %cleanup2
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
entry: unwinds to %cleanup
  br label %bb1
bb1: unwinds to %cleanup
  br label %bb2
bb2: unwinds to %cleanup
  br label %bb3
bb3:
  br label %bb4
bb4: unwinds to %cleanup
  ret i32 0
cleanup:
  ret i32 1
}

define i32 @f3() {
entry: unwinds to %cleanup
  call void @g(i32 0)
  ret i32 0
cleanup:
  unwind
}

define i32 @f4() {
entry: unwinds to %cleanup
  call void @g(i32 0)
  br label %cleanup
cleanup:
  unwind
}

define i32 @f5() {
entry: unwinds to %cleanup
  call void @g(i32 0)
  br label %other
other:
  ret i32 0
cleanup:
  unwind
}
