// RUN: clang-cc -emit-llvm %s -o %t -triple=x86_64-apple-darwin9 && 

struct T {
  T();
  ~T();
};

void f(const T& t = T());

class X { // ...
public:
        X();
        X(const X&, const T& t = T());
};

void g() {
  // RUN: grep "call void @_ZN1TC1Ev" %t | count 4 &&
  // RUN: grep "call void @_ZN1TD1Ev" %t | count 4
  f();
  f();

  X a;
  X b(a);
  X c = a;
}
