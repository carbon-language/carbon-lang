// RUN: clang-cc -emit-llvm -o - %s
struct A;
struct B;
extern A *f();
void a() { (B *) f(); }
