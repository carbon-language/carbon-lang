// RUN: %llvmgcc -c -g %s -o - | llc -O0 -o %t.s
// RUN: %compile_c %t.s -o %t.o
// PR4025

template <typename _Tp> class vector
{
public:
  ~vector ()
  {
  }
};

class Foo
{
  ~Foo();
  class FooImpl *impl_;
};

namespace {
  class Bar;
}

class FooImpl
{
  vector<Bar*> thing;
};

Foo::~Foo()
{
  delete impl_;
}

