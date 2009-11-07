// RUN: clang-cc %s -emit-llvm-only -verify

class x {
int operator=(int);
};
void a() {
  x a;
  a = 1u;
}
