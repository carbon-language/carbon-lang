// RUN: %clang_cc1 %s -emit-llvm-only

struct A { virtual void f(); };
struct B : A { };

const B& f(A *a) {
  return dynamic_cast<const B&>(*a);
}
