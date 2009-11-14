// RUN: clang-cc -emit-llvm %s -o - -triple=x86_64-apple-darwin10

// PR5484
namespace PR5484 {
struct A { };
extern A a;

void f(const A & = a);

void g() {
  f();
}
}
