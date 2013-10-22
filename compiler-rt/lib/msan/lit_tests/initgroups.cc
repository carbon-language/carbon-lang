// RUN: %clangxx_msan -m64 -O0 %s -o %t && %t

#include <sys/types.h>
#include <grp.h>

int main(void) {
  initgroups("root", 0);
  // The above fails unless you are root. Does not matter, MSan false positive
  // (which we are testing for) happens anyway.
  return 0;
}
