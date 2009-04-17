; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2

declare void @bar(i64 %x, i64 %y)

define void @foo() {
  call void @bar(i64 2, i64 3)
  ret void
}
