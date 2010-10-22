// RUN: %clang_cc1 -emit-llvm -o - %s
// rdar //8540501

struct A
{
  int i;
  A (int j) : i(j) {}
  A (const A &j) : i(j.i) {}
  A& operator= (const A &j) { i = j.i; return *this; }
};

A foo(int j)
{
  return ({ j ? A(1) : A(0); });
}

int main()
{
  return foo(1).i-1;
}

void foo2()
{
  A b = ({ A a(1); a; });
}

