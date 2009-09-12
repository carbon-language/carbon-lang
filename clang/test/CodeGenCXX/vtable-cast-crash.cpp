// RUN: clang-cc -emit-llvm-only %s
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
                         
