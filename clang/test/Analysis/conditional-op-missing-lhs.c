// RUN: %clang_cc1 -analyze -analyzer-checker=DeadStores -warn-uninit-values -verify %s
// FIXME: The current UninitializedValuesChecker will go away; replace it and re-enable test.
// XFAIL: *

void f1()
{
  int i;
  
  int j = i ? : 1; // expected-warning{{use of uninitialized variable}} //expected-warning{{Value stored to 'j' during its initialization is never read}}
}

void *f2(int *i)
{
  return i ? : 0;
}

void *f3(int *i)
{
  int a;
  
  return &a ? : i;
}

void f4()
{
  char c[1 ? : 2];
}

