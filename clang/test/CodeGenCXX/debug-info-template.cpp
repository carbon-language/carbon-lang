// RUN: %clang -emit-llvm -g -S %s -o - | FileCheck %s

//CHECK: TC<int>
//CHECK: DW_TAG_template_type_parameter

template<typename T>
class TC {
public:
  TC(const TC &) {}
  TC() {}
};

TC<int> tci;

//CHECK: TU<2>
//CHECK: DW_TAG_template_value_parameter
template<unsigned >
class TU {
  int b;
};

TU<2> u2;

// PR9600
template<typename T> class vector {};
class Foo;
typedef vector<Foo*> FooVector[3];
struct Test {
  virtual void foo(FooVector *);
};
static Test test;

