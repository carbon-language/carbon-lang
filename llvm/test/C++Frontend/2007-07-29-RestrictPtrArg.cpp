// RUN: %llvmgxx -c -emit-llvm %s -o - | llvm-dis | grep noalias
// XFAIL: i[1-9]86|alpha|ia64|arm|x86_64|amd64
// NOTE: This should be un-XFAILed when the C++ type qualifiers are fixed

void foo(int * __restrict myptr1, int * myptr2) {
  myptr1[0] = 0;
  myptr2[0] = 0;
}
