// RUN: clang-cc -emit-llvm -o - %s

extern "C" int printf(...);

class A {
public:
  A() : f(1.0), d(2.0), Ai(100) {}
  float f;
  double d;
  int Ai;
}; 

int main() 
{
  A a1;
  int A::* pa = &A::Ai;
  float A::* pf = &A::f;
  double A::* pd = &A::d;
  printf("%d %d %d\n", &A::Ai, &A::f, &A::d);

  // FIXME. NYI
  //  printf(" %d, %f, %f  \n", a1.*pa, a1.f, a1.d);
}


