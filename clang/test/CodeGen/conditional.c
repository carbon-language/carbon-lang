// RUN: clang -emit-llvm %s

float test1(int cond, float a, float b)
{
  return cond ? a : b;
}
double test2(int cond, float a, double b)
{
  return cond ? a : b;
}

void f();

void test3(){
   1 ? f() : (void)0;
}

