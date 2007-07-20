; RUN: llvm-as < %s | llc -march=x86-64 | grep movaps | wc -l | grep 1

define <2 x i64> @bar(<2 x i64>* %p)
{
  %t = load <2 x i64>* %p
  ret <2 x i64> %t
}
