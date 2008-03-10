; RUN: llvm-as < %s | opt -inline | llvm-dis | grep "br label %cleanup"

define void @g() {
  unwind
}

define i32 @f1() {
entry: unwinds to %cleanup
  call void @g()
  ret i32 0
cleanup:
  ret i32 1
}
