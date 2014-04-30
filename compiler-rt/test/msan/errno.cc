// RUN: %clangxx_msan -m64 -O0 %s -o %t && %run %t

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

int main()
{
  int x;
  int *volatile p = &x;
  errno = *p;
  int res = read(-1, 0, 0);
  assert(res == -1);
  if (errno) printf("errno %d\n", errno);
  return 0;
}
