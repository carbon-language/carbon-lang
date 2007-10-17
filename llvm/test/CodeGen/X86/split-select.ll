; RUN: llvm-as < %s | llc -march=x86-64 | grep test | count 1

define void @foo(i1 %c, <2 x float> %a, <2 x float> %b, <2 x float>* %p) {
  %x = select i1 %c, <2 x float> %a, <2 x float> %b
  store <2 x float> %x, <2 x float>* %p
  ret void
}
