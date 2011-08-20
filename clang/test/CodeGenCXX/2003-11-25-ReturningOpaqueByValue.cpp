// RUN: %clang_cc1 -emit-llvm %s -o -

// FIXME: Avoid using system headers. clang cannot handle MS <vector> yet.
// XFAIL: win32
#include <vector>
std::vector<int> my_method ();

int
main ()
{
  my_method ();
  return 0;
}
