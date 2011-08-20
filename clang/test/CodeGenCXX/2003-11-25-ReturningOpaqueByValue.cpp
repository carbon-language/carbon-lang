// RUN: %clang_cc1 -emit-llvm %s -o -

#include <vector>
std::vector<int> my_method ();

int
main ()
{
  my_method ();
  return 0;
}
