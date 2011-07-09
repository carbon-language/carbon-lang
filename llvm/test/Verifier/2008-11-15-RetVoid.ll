; RUN: not llvm-as < %s |& grep {value doesn't match function result type 'void'}

define void @foo() {
  ret i32 0
}
