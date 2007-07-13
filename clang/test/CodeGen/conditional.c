// RUN: clang -emit-llvm %s

float test1(int cond, float a, float b)
{
  return cond ? a : b;
}
/* this will fail for now...
double test2(int cond, float a, double b)
{
  return cond ? a : b;
}
*/
