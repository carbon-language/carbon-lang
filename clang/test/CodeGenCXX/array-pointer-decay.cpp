// RUN: clang-cc %s -emit-llvm -o -

void f(const char*);

void g() {
  f("hello");
}
