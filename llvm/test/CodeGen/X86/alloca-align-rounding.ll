; RUN: llc < %s -march=x86-64 -mtriple=i686-pc-linux | grep and | count 1

declare void @bar(<2 x i64>* %n)

define void @foo(i64 %h) {
  %p = alloca <2 x i64>, i64 %h
  call void @bar(<2 x i64>* %p)
  ret void
}

define void @foo2(i64 %h) {
  %p = alloca <2 x i64>, i64 %h, align 32
  call void @bar(<2 x i64>* %p)
  ret void
}
