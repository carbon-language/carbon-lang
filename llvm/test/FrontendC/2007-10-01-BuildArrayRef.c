// RUN: not %llvmgcc_only -c %s -o /dev/null |& FileCheck %s
// PR 1603
void func()
{
   const int *arr;
   arr[0] = 1;  // CXHECK: error: assignment of read-only location
}

struct foo {
  int bar;
};
struct foo sfoo = { 0 };

int func2()
{
  const struct foo *fp;
  fp = &sfoo;
  fp[0].bar = 1;  // CHECK: error: assignment of read-only member 'bar'
  return sfoo.bar;
}
