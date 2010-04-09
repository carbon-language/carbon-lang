// RUN: %clang_cc1 %s -emit-llvm-only -verify

class x {
public: int operator=(int);
};
void a() {
  x a;
  a = 1u;
}
