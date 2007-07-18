; RUN: llvm-as < %s | llc -march=x86-64 | grep movdqu | wc -l | grep 1

define void @bar(<2 x i64>* %p, <2 x i64> %x)
{
  store <2 x i64> %x, <2 x i64>* %p, align 8
  ret void
}
