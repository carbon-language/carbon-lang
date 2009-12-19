// RUN: %clang_cc1 %s -emit-llvm-only -verify

struct A {};
struct B : A {};
void a(const A& x = B());
void b() { a(); }
