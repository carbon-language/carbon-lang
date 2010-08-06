// RUN: %clang_cc1 -faltivec -fno-lax-vector-conversions -triple powerpc-unknown-unknown -verify %s

typedef int V4i __attribute__((vector_size(16)));

void f(V4i a)
{
}

void test()
{
  V4i vGCC;
  vector int vAltiVec;

  f(vAltiVec);
  vGCC = vAltiVec;
  vGCC = vGCC > vAltiVec;
  vAltiVec = 0 ? vGCC : vGCC;
}
