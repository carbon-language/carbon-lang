// RUN: %clang_cc1 -emit-llvm-only -cxx-abi itanium %s
struct A
{
A();    
virtual ~A();
};

struct B: A
{
  B();
  ~B();
};

B::B()
{
}

B::~B()
{
}
                         
