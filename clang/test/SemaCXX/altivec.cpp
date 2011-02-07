// RUN: %clang_cc1 -faltivec -fno-lax-vector-conversions -triple powerpc-unknown-unknown -verify %s

typedef int V4i __attribute__((vector_size(16)));

void f(V4i a)
{
}

void test1()
{
  V4i vGCC;
  vector int vAltiVec;

  f(vAltiVec);
  vGCC = vAltiVec;
  bool res = vGCC > vAltiVec;
  vAltiVec = 0 ? vGCC : vGCC;
}

template<typename T>
void template_f(T param) {
  param++;
}

void test2()
{
  vector int vi;
  ++vi;
  vi++;
  --vi;
  vi--;
  vector float vf;
  vf++;

  ++vi=vi;
  (++vi)[1]=1;
  template_f(vi);
}
