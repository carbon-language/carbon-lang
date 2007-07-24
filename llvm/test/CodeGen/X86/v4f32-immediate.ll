; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse | grep movaps

define <4 x float> @foo() {
  ret <4 x float> <float 3.223542354, float 2.3, float 1.2, float 0.1>
}
