// RUN: clang-cc -emit-llvm %s -o %t -triple=x86_64-apple-darwin9 && 

struct T {
  T();
  ~T();
};

void f(const T& t = T());

void g() {
  // RUN: grep "call void @_ZN1TC1Ev" %t | count 2 &&
  // RUN: grep "call void @_ZN1TD1Ev" %t | count 2
  f();
  f();
}
