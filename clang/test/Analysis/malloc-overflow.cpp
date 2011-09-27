// RUN: %clang_cc1 -analyze -analyzer-checker=experimental.security.MallocOverflow -verify %s

class A {
public:
  A& operator<<(const A &a);
};

void f() {
  A a = A(), b = A();
  a << b;
}
