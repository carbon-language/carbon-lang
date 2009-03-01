; RUN: llvm-as < %s | llvm-dis

type i32

define void @foo() {
  bitcast %0* null to i32*
  ret void
}
